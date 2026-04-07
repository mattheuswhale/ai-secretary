"""
mcp_host.py — MCP client / tool orchestrator
Manages a pool of stdio MCP server subprocesses and exposes their
tools as Ollama-compatible function specs to the voice assistant.

Security notes:
  - stdio transport only — avoids CVE-2025-66416 DNS-rebinding attack
    surface entirely (HTTP MCP servers on localhost are vulnerable).
  - Servers run as unprivileged subprocesses; no ports are opened.
  - Tool arguments validated with JSON Schema before dispatch.
  - Output size capped before returning to LLM context.
"""

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

# pip install mcp>=1.23.0
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

log = logging.getLogger("mcp_host")

# Maximum characters a single tool result may return to the LLM.
# Prevents context-window flooding from runaway tool output.
TOOL_OUTPUT_MAX_CHARS = int(os.getenv("TOOL_OUTPUT_MAX_CHARS", "8000"))


# ═══════════════════════════════════════════════════════════════════════
# SERVER REGISTRY
# Each entry describes one MCP server subprocess.
# Add your own servers here or load from mcp_config.json.
#
# Security: only list servers you trust. An MCP server runs as your
# user and can do anything your user can do on the host machine.
# ═══════════════════════════════════════════════════════════════════════

def load_server_registry(config_path: str = "mcp_config.json") -> list[dict]:
    """
    Load server definitions from mcp_config.json if it exists,
    otherwise fall back to the hardcoded example registry.
    """
    p = Path(config_path)
    if p.exists():
        try:
            with open(p) as f:
                cfg = json.load(f)
            servers = cfg.get("servers", [])
            log.info(f"[mcp_host] Loaded {len(servers)} server(s) from {config_path}")
            return servers
        except Exception as e:
            log.error(f"[mcp_host] Failed to load {config_path}: {e}")

    # Hardcoded fallback — replace with your own servers
    return [
        {
            "name": "documents",
            "command": "python",
            "args": ["agents/documents_agent.py"],
            "env": {},           # extra env vars for this subprocess
            "enabled": True,
        },
        {
            "name": "rss",
            "command": "python",
            "args": ["agents/rss_agent.py"],
            "env": {},
            "enabled": True,
        },
        # Add more servers here:
        # {
        #     "name": "code",
        #     "command": "python",
        #     "args": ["agents/code_agent.py"],
        #     "env": {},
        #     "enabled": False,   # set True to activate
        # },
    ]


# ═══════════════════════════════════════════════════════════════════════
# MCPHost — manages server lifecycle and tool dispatch
# ═══════════════════════════════════════════════════════════════════════

class MCPHost:
    def __init__(self):
#        self._sessions: dict[str, ClientSession] = {}
#        self._tool_index: dict[str, tuple[str, dict]] = {}  # tool_name → (server_name, schema)
#        self._registry = load_server_registry()

        self._sessions: dict[str, ClientSession] = {}
        self._tool_index: dict[str, tuple[str, dict]] = {}
        self._registry = load_server_registry()

        self._stop_event = asyncio.Event()
        self._ready_event = asyncio.Event()
        self._tg: asyncio.TaskGroup | None = None
        self._pending_servers = 0

    # ── Lifecycle ───────────────────────────────────────────────────────

#    async def start(self) -> None:
#        """Spawn all enabled servers and enumerate their tools."""
#        for server_def in self._registry:
#            if not server_def.get("enabled", True):
#                continue
#            name = server_def["name"]
#            try:
#                await self._connect_server(name, server_def)
#            except Exception as e:
#                log.error(f"[mcp_host] Failed to start server '{name}': {e}")
#
#        n_tools = len(self._tool_index)
#        n_servers = len(self._sessions)
#        log.info(f"[mcp_host] Ready — {n_servers} server(s), {n_tools} tool(s)")

    async def start(self):
        self._stop_event.clear()
        self._ready_event.clear()

        self._pending_servers = sum(
            1 for s in self._registry if s.get("enabled", True)
        )

        self._tg = asyncio.TaskGroup()
        await self._tg.__aenter__()

        for server_def in self._registry:
            if not server_def.get("enabled", True):
                continue
            name = server_def["name"]
            self._tg.create_task(self._connect_server(name, server_def))

        # wait until all servers are ready
        await self._ready_event.wait()


#    async def _stop_all(self):
#        for name, (session, cm) in list(self._sessions.items()):
#            try:
#                await session.__aexit__(None, None, None)
#            except Exception:
#                pass
#            try:
#                await cm.__aexit__(None, None, None)
#            except Exception:
#                pass
    async def _stop_all(self):
        self._stop_event.set()

        if self._tg:
            await self._tg.__aexit__(None, None, None)
            self._tg = None


    async def stop(self) -> None:
        """Close all server sessions."""

        for name, (session, cm) in list(self._sessions.items()):
            try:
                await session.__aexit__(None, None, None)
            except Exception:
                pass

            try:
                await cm.__aexit__(None, None, None)
            except Exception:
                pass


#        for name, session in list(self._sessions.items()):
#            try:
#                await session.__aexit__(None, None, None)
#            except Exception:
#                pass
        self._sessions.clear()
        self._tool_index.clear()

#    async def _connect_server(self, name: str, server_def: dict) -> None:
#        """Connect to a single stdio MCP server and index its tools."""
#        # Build a clean env for the subprocess — inherit current env,
#        # then layer in any server-specific overrides.
#        env = {**os.environ, **server_def.get("env", {})}
#
#        params = StdioServerParameters(
#            command=server_def["command"],
#            args=server_def.get("args", []),
#            env=env,
#        )
#
#        # stdio_client returns an async context manager
#
#        cm = stdio_client(params)
#        read, write = await cm.__aenter__()
#
#        session = ClientSession(read, write)
#        await session.__aenter__()
#
#        # store BOTH session and context manager
#        self._sessions[name] = (session, cm)
#
#
#        #read, write = await stdio_client(params).__aenter__()
#        #session = ClientSession(read, write)
#        #await session.__aenter__()
#
#        await session.initialize()
#
#        self._sessions[name] = session
#
#        # Enumerate and index all tools this server exposes
#        tools_response = await session.list_tools()
#        for tool in tools_response.tools:
#            if tool.name in self._tool_index:
#                log.warning(
#                    f"[mcp_host] Tool name collision: '{tool.name}' "
#                    f"already registered by '{self._tool_index[tool.name][0]}'. "
#                    f"Server '{name}' will override it."
#                )
#            self._tool_index[tool.name] = (name, tool.inputSchema or {})
#            log.debug(f"[mcp_host] Registered tool '{tool.name}' from server '{name}'")

    async def _connect_server(self, name: str, server_def: dict) -> None:
        env = {**os.environ, **server_def.get("env", {})}
        params = StdioServerParameters(
            command=server_def["command"],
            args=server_def.get("args", []),
            env=env,
        )

        try:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._sessions[name] = session

                    # index tools
                    tools_response = await session.list_tools()
                    for tool in tools_response.tools:
                        self._tool_index[tool.name] = (
                            name,
                            tool.inputSchema or {},
                        )

                    # signal readiness
                    self._pending_servers -= 1
                    if self._pending_servers == 0:
                        self._ready_event.set()

                    # stay alive until shutdown
                    await self._stop_event.wait()

        except Exception as e:
            log.error(f"[mcp_host] Server '{name}' crashed: {e}")

        finally:
            self._sessions.pop(name, None)


    # ── Tool access ─────────────────────────────────────────────────────

    def get_ollama_tools(self) -> list[dict]:
        """
        Return all registered tools formatted for Ollama's tool-calling API.
        Called by LLMAgent to build its payload.
        """
        specs = []
        for tool_name, (server_name, schema) in self._tool_index.items():
            specs.append({
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": schema.get("description", f"Tool '{tool_name}' from {server_name}"),
                    "parameters": {
                        "type": "object",
                        "properties": schema.get("properties", {}),
                        "required": schema.get("required", []),
                    },
                },
            })
        return specs

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """
        Dispatch a tool call to the appropriate MCP server.
        Returns a JSON string suitable for the Ollama tool-result role.
        """
        if tool_name not in self._tool_index:
            return json.dumps({"error": f"Unknown tool: {tool_name!r}"})

        server_name, schema = self._tool_index[tool_name]

#        session = self._sessions.get(server_name)

        entry = self._sessions.get(server_name)
        if entry is None:
            return json.dumps({"error": f"Server '{server_name}' is not running"})

        session = entry

        if session is None:
            return json.dumps({"error": f"Server '{server_name}' is not running"})

        # Validate argument types against the JSON schema before dispatch.
        # Prevents malformed LLM output from reaching the tool implementation.
        validation_error = _validate_arguments(arguments, schema)
        if validation_error:
            return json.dumps({"error": f"Argument validation failed: {validation_error}"})

        try:
            result = await session.call_tool(tool_name, arguments)
            # MCP returns a list of content blocks; join text blocks.
            text_parts = [
                block.text for block in result.content
                if hasattr(block, "text")
            ]
            output = "\n".join(text_parts)

            # Cap output size — prevents LLM context flooding
            if len(output) > TOOL_OUTPUT_MAX_CHARS:
                output = output[:TOOL_OUTPUT_MAX_CHARS] + f"\n[output truncated at {TOOL_OUTPUT_MAX_CHARS} chars]"

            return json.dumps({"result": output}, ensure_ascii=False)

        except Exception as e:
            log.error(f"[mcp_host] Tool '{tool_name}' raised: {e}")
            return json.dumps({"error": f"Tool execution error: {type(e).__name__}"})

    def list_tools(self) -> list[str]:
        return list(self._tool_index.keys())


# ═══════════════════════════════════════════════════════════════════════
# Argument validator — lightweight JSON Schema subset check
# Covers string, number, integer, boolean, array, object types.
# Full JSON Schema validation would require jsonschema library.
# ═══════════════════════════════════════════════════════════════════════

_TYPE_MAP = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
}

def _validate_arguments(args: dict, schema: dict) -> str | None:
    """
    Returns an error string if validation fails, None if OK.
    Only validates required fields and basic type checking.
    """
    if not schema:
        return None

    properties = schema.get("properties", {})
    required = schema.get("required", [])

    for field in required:
        if field not in args:
            return f"Missing required field: '{field}'"

    for field, value in args.items():
        if field not in properties:
            continue  # extra fields are allowed
        expected_type = properties[field].get("type")
        if expected_type and expected_type in _TYPE_MAP:
            if not isinstance(value, _TYPE_MAP[expected_type]):
                return (
                    f"Field '{field}': expected {expected_type}, "
                    f"got {type(value).__name__}"
                )
        # String length cap — prevents injection via oversized inputs
        if isinstance(value, str) and len(value) > 4000:
            return f"Field '{field}': string too long (max 4000 chars)"

    return None


# ═══════════════════════════════════════════════════════════════════════
# Sync wrapper — voice assistant main loop is synchronous
# ═══════════════════════════════════════════════════════════════════════

class MCPHostSync:
    """
    Thin synchronous wrapper around MCPHost for use from synchronous code.
    Runs an event loop in a background thread.
    """
    def __init__(self):
        import threading
        self._loop = asyncio.new_event_loop()
        self._host = MCPHost()
        self._thread = threading.Thread(
            target=self._loop.run_forever, daemon=True, name="mcp-event-loop"
        )
        self._thread.start()

    def start(self) -> None:
#        future = asyncio.run_coroutine_threadsafe(self._host.start(), self._loop)
#        future.result(timeout=30)
        future = asyncio.run_coroutine_threadsafe(
            self._host.start(), self._loop
        )
        future.result(timeout=10)

    def stop(self) -> None:
#        #future = asyncio.run_coroutine_threadsafe(self._host.stop(), self._loop)
#        #future.result(timeout=10)
#        future = asyncio.run_coroutine_threadsafe(self._host._stop_all(), self._loop)
#        future.result(timeout=10)
        future = asyncio.run_coroutine_threadsafe(
            self._host._stop_all(), self._loop
        )
        future.result(timeout=10)

    def get_ollama_tools(self) -> list[dict]:
        return self._host.get_ollama_tools()

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        future = asyncio.run_coroutine_threadsafe(
            self._host.call_tool(tool_name, arguments), self._loop
        )
        return future.result(timeout=30)

    def list_tools(self) -> list[str]:
        return self._host.list_tools()
