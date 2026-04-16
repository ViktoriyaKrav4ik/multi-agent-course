"""
ReportMCP Server — save_report tool.

Порт: 8902 (за замовчуванням).
Використовується Supervisor для збереження звітів (через HITL).

Запуск: python mcp_servers/report_mcp.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastmcp import FastMCP

from config import Settings

settings = Settings()
mcp = FastMCP(name="ReportMCP")

_ROOT = Path(__file__).resolve().parent.parent
_OUTPUT_DIR = _ROOT / settings.output_dir


# ── Resources ───────────────────────────────────────────────────

@mcp.resource("resource://output-dir")
def output_dir_info() -> str:
    """Шлях до директорії збережених звітів та список файлів."""
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(
        [f.name for f in _OUTPUT_DIR.iterdir() if f.is_file() and f.suffix == ".md"]
    )
    return json.dumps(
        {
            "output_dir": str(_OUTPUT_DIR),
            "reports": files,
            "total": len(files),
        },
        ensure_ascii=False,
    )


# ── Tools ───────────────────────────────────────────────────────

@mcp.tool()
def save_report(filename: str, content: str) -> str:
    """Зберегти Markdown-звіт у output/. Потребує підтвердження користувача (HITL на Supervisor)."""
    if not filename.strip():
        return "Помилка: вкажіть ім'я файлу."
    filename = filename.strip()
    if not filename.endswith(".md"):
        filename = filename + ".md"
    if filename.lower() == "report.md":
        filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = _OUTPUT_DIR / filename

    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Звіт збережено: {path}"
    except Exception as e:
        return f"Помилка збереження файлу: {e}"


# ── Entrypoint ──────────────────────────────────────────────────

if __name__ == "__main__":
    port = settings.report_mcp_port
    print(f"📄 ReportMCP starting on port {port}...")
    mcp.run(transport="streamable-http", host="127.0.0.1", port=port)
