"""
agents/agent_template.py — Copy this file to create a new MCP server agent.

Steps:
  1. Copy: cp agents/agent_template.py agents/my_agent.py
  2. Rename the FastMCP instance and update `instructions`.
  3. Replace the example tools with your own @mcp.tool() functions.
  4. Add an entry to mcp_config.json (or the registry in mcp_host.py).
  5. Set "enabled": true in the config and restart the voice assistant.

Security checklist for every new tool:
  [ ] Never pass user/LLM input directly to shell commands (subprocess, os.system).
      Use shlex.quote() if shell is unavoidable, or prefer Python-native APIs.
  [ ] Validate and sanitize all string inputs (strip, length cap, allowlist chars).
  [ ] For file operations: resolve paths and check against a root directory.
  [ ] For network requests: allowlist domains; set timeouts; cap response size.
  [ ] Never return secrets, credentials, or full stack traces to the LLM.
  [ ] Keep tool output under TOOL_OUTPUT_MAX_CHARS (set in mcp_host.py).
"""

import json
import os
from mcp.server.fastmcp import FastMCP

# ── Config ───────────────────────────────────────────────────────────────
# Load any agent-specific settings from environment variables (set in .env
# or passed via the "env" field in mcp_config.json).
EXAMPLE_SETTING = os.getenv("EXAMPLE_SETTING", "default_value")


# ── FastMCP server ───────────────────────────────────────────────────────
mcp = FastMCP(
    "my_agent",           # ← change to your agent's name
    instructions=(
        "One sentence describing what this agent does and when to use it. "
        "This text appears in the LLM's context when tools are listed."
    ),
)


# ── Tools ────────────────────────────────────────────────────────────────

@mcp.tool()
def example_tool(input_text: str, optional_flag: bool = False) -> str:
    """
    One-line description used as the tool description in the LLM.
    Keep it concise — the LLM uses this to decide whether to call the tool.

    Args:
        input_text: Description of this parameter (shown to LLM).
        optional_flag: Description of this parameter.
    """
    # ── Input validation ─────────────────────────────────────────────────
    if not isinstance(input_text, str) or not input_text.strip():
        return json.dumps({"error": "input_text must be a non-empty string"})
    input_text = input_text.strip()[:500]  # length cap

    # ── Your logic here ──────────────────────────────────────────────────
    result = f"Processed: {input_text!r}, flag={optional_flag}"

    return json.dumps({"result": result}, ensure_ascii=False)


@mcp.tool()
def another_tool(query: str) -> str:
    """
    Another tool example. Add as many as your agent needs.

    Args:
        query: The search or query string.
    """
    if not query or not query.strip():
        return json.dumps({"error": "query cannot be empty"})

    # TODO: implement your tool logic here
    return json.dumps({"result": f"Results for: {query!r}"}, ensure_ascii=False)


# ── Entry point ──────────────────────────────────────────────────────────
# Transport is always stdio — do not change this.
# stdio transport is immune to the DNS-rebinding CVE that affects HTTP servers.

if __name__ == "__main__":
    import logging, sys
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    mcp.run(transport="stdio")
