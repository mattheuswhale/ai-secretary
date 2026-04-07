"""
agents/rss_agent.py — RSS feed MCP server

Bridges the voice assistant to your existing FreshRSS server.
Exposes tools to fetch, list, and summarize RSS feed items.

Reads connection settings from environment variables (set in .env):
    FRESHRSS_URL      — base URL of your FreshRSS instance
    FRESHRSS_USER     — API username
    FRESHRSS_API_KEY  — GReader API password / token
"""

import json
import os
import re
import time
from datetime import datetime, timezone

import requests
from mcp.server.fastmcp import FastMCP

# ── Config ───────────────────────────────────────────────────────────────
FRESHRSS_URL     = os.getenv("FRESHRSS_URL",     "http://localhost/freshrss")
FRESHRSS_USER    = os.getenv("FRESHRSS_USER",    "your_username")
FRESHRSS_API_KEY = os.getenv("FRESHRSS_API_KEY", "your_api_or_pasword")

FETCH_TIMEOUT    = int(os.getenv("RSS_FETCH_TIMEOUT",   "10"))
MAX_ITEMS        = int(os.getenv("RSS_MAX_ITEMS",        "20"))
SNIPPET_MAX_CHARS= int(os.getenv("RSS_SNIPPET_MAX_CHARS","400"))


# ── FreshRSS GReader API client ──────────────────────────────────────────

class FreshRSSClient:
    def __init__(self):
        self._token: str | None = None
        self._token_expiry: float = 0.0

    def _auth_header(self) -> dict:
        if not FRESHRSS_USER or not FRESHRSS_API_KEY:
            return {}
        now = time.time()
        if self._token is None or now > self._token_expiry:
            self._token = self._get_token()
            self._token_expiry = now + 3500  # tokens valid ~1 hour
        return {"Authorization": f"GoogleLogin auth={self._token}"}

    def _get_token(self) -> str:
        url = f"{FRESHRSS_URL}/api/greader.php/accounts/ClientLogin"
        r = requests.post(
            url,
            data={"Email": FRESHRSS_USER, "Passwd": FRESHRSS_API_KEY},
            timeout=FETCH_TIMEOUT,
            verify=True,
        )
        r.raise_for_status()
        for line in r.text.splitlines():
            if line.startswith("Auth="):
                return line[5:].strip()
        raise ValueError("No Auth token in FreshRSS response")

    def get_unread(self, n: int = MAX_ITEMS) -> list[dict]:
        url = f"{FRESHRSS_URL}/api/greader.php/reader/api/0/stream/contents/user/-/state/com.google/reading-list"
        r = requests.get(
            url,
            params={"n": n, "xt": "user/-/state/com.google/read"},
            headers=self._auth_header(),
            timeout=FETCH_TIMEOUT,
            verify=True,
        )
        r.raise_for_status()
        items = r.json().get("items", [])
        return [self._clean_item(i) for i in items]

    def get_items_by_feed(self, feed_title: str, n: int = MAX_ITEMS) -> list[dict]:
        # Fetch unread and filter by feed title (case-insensitive)
        all_items = self.get_unread(n=min(n * 5, 100))
        pattern = re.compile(re.escape(feed_title.strip()), re.IGNORECASE)
        return [i for i in all_items if pattern.search(i.get("feed", ""))][:n]

    def mark_read(self, item_id: str) -> bool:
        url = f"{FRESHRSS_URL}/api/greader.php/reader/api/0/edit-tag"
        r = requests.post(
            url,
            data={"i": item_id, "a": "user/-/state/com.google/read"},
            headers=self._auth_header(),
            timeout=FETCH_TIMEOUT,
            verify=True,
        )
        return r.status_code == 200

    @staticmethod
    def _clean_item(raw: dict) -> dict:
        published = raw.get("published", 0)
        ts = datetime.fromtimestamp(published, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if published else "unknown"
        summary_html = (raw.get("summary") or {}).get("content", "") or ""
        # Strip HTML tags for LLM consumption
        summary_text = re.sub(r"<[^>]+>", "", summary_html).strip()
        return {
            "id":      raw.get("id", ""),
            "title":   raw.get("title", "(no title)"),
            "feed":    (raw.get("origin") or {}).get("title", ""),
            "url":     (raw.get("alternate") or [{}])[0].get("href", ""),
            "date":    ts,
            "snippet": summary_text[:SNIPPET_MAX_CHARS],
        }


_client = FreshRSSClient()


# ── FastMCP server ───────────────────────────────────────────────────────

mcp = FastMCP(
    "rss",
    instructions=(
        "Access the user's FreshRSS feed reader. "
        "Use get_unread_articles to fetch new items, "
        "get_articles_from_feed to filter by feed name, "
        "and mark_article_read to mark items as read."
    ),
)


@mcp.tool()
def get_unread_articles(max_items: int = 10) -> str:
    """
    Fetch unread RSS articles from FreshRSS.

    Args:
        max_items: Maximum number of articles to return (1-20).
    """
    max_items = max(1, min(int(max_items), MAX_ITEMS))
    try:
        items = _client.get_unread(n=max_items)
        return json.dumps({"count": len(items), "articles": items}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"FreshRSS error: {type(e).__name__}: {e}"})


@mcp.tool()
def get_articles_from_feed(feed_name: str, max_items: int = 10) -> str:
    """
    Fetch unread articles from a specific RSS feed.

    Args:
        feed_name: Feed title to filter by (partial match, case-insensitive).
        max_items: Maximum number of articles to return.
    """
    if not feed_name or not feed_name.strip():
        return json.dumps({"error": "feed_name cannot be empty"})
    feed_name = feed_name.strip()[:100]
    max_items = max(1, min(int(max_items), MAX_ITEMS))

    try:
        items = _client.get_items_by_feed(feed_name, n=max_items)
        return json.dumps({"feed": feed_name, "count": len(items), "articles": items}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"FreshRSS error: {type(e).__name__}: {e}"})


@mcp.tool()
def mark_article_read(article_id: str) -> str:
    """
    Mark an RSS article as read.

    Args:
        article_id: The article ID returned by get_unread_articles.
    """
    if not article_id or not article_id.strip():
        return json.dumps({"error": "article_id cannot be empty"})
    article_id = article_id.strip()[:500]  # GReader IDs can be long URIs

    try:
        ok = _client.mark_read(article_id)
        return json.dumps({"success": ok, "article_id": article_id})
    except Exception as e:
        return json.dumps({"error": f"FreshRSS error: {type(e).__name__}"})


if __name__ == "__main__":
    import logging, sys
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    mcp.run(transport="stdio")
