"""
agents/documents_agent.py — Document folder MCP server (template)

Run standalone for testing:
    python agents/documents_agent.py

Exposes tools:
    list_documents   — list files in the document folder
    read_document    — read a file's content
    search_documents — grep for a term across files
    write_document   — create or overwrite a file (with confirmation flag)

Security:
    - All paths are resolved and checked against DOCUMENTS_ROOT to
      prevent path traversal (../../etc/passwd style attacks).
    - File size cap on reads to prevent memory exhaustion.
    - write_document requires confirm=True to prevent accidental overwrites.
    - Shell commands are never used — all ops use Python stdlib.
"""

import json
import os
import re
from pathlib import Path

# pip install mcp>=1.23.0
from mcp.server.fastmcp import FastMCP

# ── Config ───────────────────────────────────────────────────────────────
DOCUMENTS_ROOT = Path(
    os.getenv("DOCUMENTS_ROOT", str(Path.home() / "Documents"))
).resolve()

READ_MAX_BYTES  = int(os.getenv("DOC_READ_MAX_BYTES",  str(1024 * 512)))   # 512 KB
SEARCH_MAX_HITS = int(os.getenv("DOC_SEARCH_MAX_HITS", "20"))

ALLOWED_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst",
    ".py", ".js", ".ts", ".json", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".csv", ".log",
}

# ── Security helper ──────────────────────────────────────────────────────

def _safe_path(filename: str) -> Path | None:
    """
    Resolve a user-supplied filename against DOCUMENTS_ROOT.
    Returns None if the resolved path escapes the root (path traversal).
    """
    # Strip leading slashes / dots to prevent absolute path injection
    clean = re.sub(r"[^\w\s\-\./]", "", filename).strip().lstrip("/")
    if not clean:
        return None
    resolved = (DOCUMENTS_ROOT / clean).resolve()
    try:
        resolved.relative_to(DOCUMENTS_ROOT)  # raises ValueError if outside root
    except ValueError:
        return None
    return resolved


# ── FastMCP server ───────────────────────────────────────────────────────
# stdio transport is used implicitly when run as a subprocess — this is
# intentional and avoids the DNS-rebinding CVE (CVE-2025-66416) that
# affects HTTP-based MCP servers on localhost.

mcp = FastMCP(
    "documents",
    instructions=(
        "Manages the user's Documents folder. "
        "Use list_documents to browse, read_document to read, "
        "search_documents to search, write_document to create/update files."
    ),
)


@mcp.tool()
def list_documents(subfolder: str = "") -> str:
    """
    List files in the documents folder.

    Args:
        subfolder: Optional subfolder path relative to DOCUMENTS_ROOT.
    """
    target = _safe_path(subfolder) if subfolder else DOCUMENTS_ROOT
    if target is None:
        return json.dumps({"error": "Invalid path"})
    if not target.exists():
        return json.dumps({"error": f"Folder not found: {subfolder!r}"})
    if not target.is_dir():
        return json.dumps({"error": f"Not a directory: {subfolder!r}"})

    entries = []
    for item in sorted(target.iterdir()):
        if item.name.startswith("."):
            continue  # skip hidden files
        entries.append({
            "name": item.name,
            "type": "dir" if item.is_dir() else "file",
            "size": item.stat().st_size if item.is_file() else None,
            "extension": item.suffix if item.is_file() else None,
        })

    return json.dumps({"folder": str(target), "entries": entries}, ensure_ascii=False)


@mcp.tool()
def read_document(filename: str) -> str:
    """
    Read the content of a document.

    Args:
        filename: File path relative to DOCUMENTS_ROOT.
    """
    path = _safe_path(filename)
    if path is None:
        return json.dumps({"error": "Invalid or unsafe path"})
    if not path.exists():
        return json.dumps({"error": f"File not found: {filename!r}"})
    if not path.is_file():
        return json.dumps({"error": f"Not a file: {filename!r}"})
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        return json.dumps({"error": f"File type not allowed: {path.suffix!r}"})

    size = path.stat().st_size
    if size > READ_MAX_BYTES:
        return json.dumps({
            "error": f"File too large ({size} bytes, max {READ_MAX_BYTES}). "
                     f"Use search_documents to find specific content."
        })

    try:
        content = path.read_text(encoding="utf-8", errors="replace")
        return json.dumps({
            "filename": filename,
            "size_bytes": size,
            "content": content,
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Read error: {type(e).__name__}"})


@mcp.tool()
def search_documents(term: str, subfolder: str = "", file_extension: str = "") -> str:
    """
    Search for a term across text files in the documents folder.

    Args:
        term: Search term (plain text, case-insensitive).
        subfolder: Optional subfolder to limit search scope.
        file_extension: Optional extension filter e.g. ".md".
    """
    if not term or not term.strip():
        return json.dumps({"error": "Search term cannot be empty"})

    # Sanitize search term — no regex injection
    safe_term = re.escape(term.strip()[:200])
    pattern = re.compile(safe_term, re.IGNORECASE)

    target = _safe_path(subfolder) if subfolder else DOCUMENTS_ROOT
    if target is None:
        return json.dumps({"error": "Invalid path"})

    ext_filter = file_extension.lower() if file_extension else None
    hits = []

    for path in sorted(target.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        if path.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue
        if ext_filter and path.suffix.lower() != ext_filter:
            continue
        if path.stat().st_size > READ_MAX_BYTES:
            continue

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        for i, line in enumerate(content.splitlines(), start=1):
            if pattern.search(line):
                hits.append({
                    "file": str(path.relative_to(DOCUMENTS_ROOT)),
                    "line": i,
                    "preview": line.strip()[:200],
                })
                if len(hits) >= SEARCH_MAX_HITS:
                    return json.dumps({
                        "term": term,
                        "hits": hits,
                        "truncated": True,
                    }, ensure_ascii=False)

    return json.dumps({"term": term, "hits": hits, "truncated": False}, ensure_ascii=False)


@mcp.tool()
def write_document(filename: str, content: str, confirm: bool = False) -> str:
    """
    Write content to a file in the documents folder.
    Requires confirm=True as a safety gate against accidental overwrites.

    Args:
        filename: File path relative to DOCUMENTS_ROOT.
        content: Text content to write.
        confirm: Must be True to actually write. If False, returns a preview.
    """
    path = _safe_path(filename)
    if path is None:
        return json.dumps({"error": "Invalid or unsafe path"})
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        return json.dumps({"error": f"File type not allowed: {path.suffix!r}"})
    if len(content) > READ_MAX_BYTES:
        return json.dumps({"error": f"Content too large (max {READ_MAX_BYTES} bytes)"})

    exists = path.exists()

    if not confirm:
        return json.dumps({
            "preview": True,
            "filename": filename,
            "action": "overwrite" if exists else "create",
            "content_length": len(content),
            "message": "Call again with confirm=True to proceed.",
        })

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return json.dumps({
            "success": True,
            "filename": filename,
            "action": "overwritten" if exists else "created",
            "bytes_written": len(content.encode("utf-8")),
        })
    except Exception as e:
        return json.dumps({"error": f"Write error: {type(e).__name__}"})


# ── Entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING, stream=__import__("sys").stderr)
    mcp.run(transport="stdio")
