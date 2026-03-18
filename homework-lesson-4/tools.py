from __future__ import annotations

import os
import re
from contextlib import redirect_stderr
from datetime import datetime
from io import StringIO
from typing import Any

from config import Settings

settings = Settings()


def tool_definitions() -> list[dict[str, Any]]:
    """
    Tools in OpenAI tool-calling JSON schema format.
    Provider-agnostic idea: name + JSON Schema params.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for a query and return top results with title, url, snippet.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query."},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_url",
                "description": "Fetch and extract the main text content from a URL (articles, docs).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "http(s) URL to read."},
                    },
                    "required": ["url"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_report",
                "description": "Save a Markdown report to disk and return the saved path.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Output filename (topic_based.md).",
                        },
                        "content": {
                            "type": "string",
                            "description": "Full Markdown content to write.",
                        },
                    },
                    "required": ["filename", "content"],
                    "additionalProperties": False,
                },
            },
        },
    ]


def web_search(query: str) -> str:
    try:
        from ddgs import DDGS
    except ImportError:
        return "ToolError: ddgs is not installed. Run: pip install ddgs"

    try:
        with redirect_stderr(StringIO()):
            results = DDGS().text(query, max_results=settings.max_search_results)
    except Exception as e:
        return f"ToolError: search failed: {e}"

    if not results:
        return "No results."

    lines: list[str] = []
    for i, r in enumerate(results, 1):
        title = (r.get("title") or "").strip()
        href = (r.get("href") or "").strip()
        body = (r.get("body") or "").strip()
        lines.append(f"{i}. {title}\n   URL: {href}\n   {body}")
    return "\n\n".join(lines)


def read_url(url: str) -> str:
    try:
        import trafilatura
    except ImportError:
        return "ToolError: trafilatura is not installed. Run: pip install trafilatura"

    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        return "ToolError: invalid URL (must start with http:// or https://)."

    try:
        downloaded = trafilatura.fetch_url(url)
    except Exception as e:
        return f"ToolError: fetch failed: {e}"

    if not downloaded:
        return "ToolError: failed to download page (timeout/404/unavailable)."

    text = trafilatura.extract(downloaded)
    if not text or not text.strip():
        return "ToolError: failed to extract readable text."

    max_len = settings.max_url_content_length
    if len(text) > max_len:
        text = text[:max_len] + "\n\n[... truncated ...]"
    return text


_FILENAME_SAFE_RE = re.compile(r"[^a-z0-9_]+")


def _sanitize_filename(filename: str) -> str:
    name = (filename or "").strip().lower()
    name = name.replace(".md", "")
    name = name.replace("-", "_").replace(" ", "_")
    name = _FILENAME_SAFE_RE.sub("_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        name = "report"
    return f"{name}.md"


def write_report(filename: str, content: str) -> str:
    filename = _sanitize_filename(filename)
    if filename == "report.md":
        filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

    root = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(root, settings.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)

    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content or "")
        return f"Report saved: {path}"
    except Exception as e:
        return f"ToolError: failed to write file: {e}"


TOOL_REGISTRY: dict[str, Any] = {
    "web_search": web_search,
    "read_url": read_url,
    "write_report": write_report,
}

