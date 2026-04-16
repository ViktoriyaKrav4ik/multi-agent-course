"""
REPL для Supervisor з Human-in-the-Loop на save_report.

Supervisor делегує агентам через ACP, зберігає звіти через ReportMCP.
Весь REPL — async, щоб коректно викликати async tools (ACP/MCP клієнти).
"""

from __future__ import annotations

import asyncio
import json
import uuid

from langchain_core.messages import HumanMessage
from langgraph.types import Command, Interrupt

from config import Settings
from supervisor import supervisor

settings = Settings()


def _print_interrupt(interrupt: Interrupt) -> None:
    print("\n" + "=" * 60)
    print("⏸️  ПОТРІБНЕ ПІДТВЕРДЖЕННЯ (save_report)")
    print("=" * 60)
    val = getattr(interrupt, "value", None) or {}
    for req in val.get("action_requests", []):
        if not isinstance(req, dict):
            print(f"  Request:  {req!r}")
            continue
        tool_name = req.get("name") or req.get("action") or "N/A"
        print(f"  Tool:  {tool_name}")
        args = req.get("args") or req.get("arguments") or {}
        preview = json.dumps(args, ensure_ascii=False, indent=2)
        if len(preview) > 4000:
            preview = preview[:4000] + "\n  ... [обрізано] ..."
        print(f"  Args:  {preview}")
    print()


def _find_interrupt(obj: object) -> Interrupt | None:
    if isinstance(obj, Interrupt):
        return obj
    if isinstance(obj, dict):
        if "__interrupt__" in obj and isinstance(obj["__interrupt__"], Interrupt):
            return obj["__interrupt__"]
        for v in obj.values():
            found = _find_interrupt(v)
            if found is not None:
                return found
    if isinstance(obj, (list, tuple)):
        for item in obj:
            found = _find_interrupt(item)
            if found is not None:
                return found
    return None


def _extract_interrupt_from_step(step: object) -> Interrupt | None:
    return _find_interrupt(step)


async def run_turn(user_text: str, thread_id: str) -> None:
    cfg = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": settings.max_iterations * 6 + 24,
    }
    stream_input: dict | Command = {"messages": [HumanMessage(content=user_text)]}

    while True:
        pending: Interrupt | None = None
        last_assistant = None
        async for step in supervisor.astream(stream_input, cfg):
            ir = _extract_interrupt_from_step(step)
            if ir is not None:
                pending = ir
            if isinstance(step, dict):
                for upd in step.values():
                    if isinstance(upd, dict) and "messages" in upd:
                        for m in upd["messages"]:
                            last_assistant = m

        if pending is None:
            if last_assistant is not None:
                content = getattr(last_assistant, "content", None) or getattr(
                    last_assistant, "text", None
                )
                if content:
                    print(f"\nAgent: {content}")
            break

        _print_interrupt(pending)
        choice = input("👉 approve / edit / reject: ").strip().lower()

        if choice == "approve":
            resume_payload = {"decisions": [{"type": "approve"}]}
        elif choice == "reject":
            reason = input("Причина (опційно): ").strip() or "скасовано користувачем"
            resume_payload = {"decisions": [{"type": "reject", "message": reason}]}
        elif choice == "edit":
            feedback = input("✏️  Що змінити у звіті перед збереженням: ").strip()
            resume_payload = {
                "decisions": [
                    {
                        "type": "edit",
                        "edited_action": {"feedback": feedback},
                    }
                ]
            }
        else:
            print("Невідома команда — введіть approve, edit або reject.")
            resume_payload = {"decisions": [{"type": "reject", "message": "скасовано"}]}

        iid = getattr(pending, "id", None)
        if iid is not None:
            stream_input = Command(resume={iid: resume_payload})
        else:
            stream_input = Command(resume=resume_payload)


async def main() -> None:
    print("Multi-agent Supervisor (hw9 — MCP + ACP). Команди: exit / quit")
    print(f"  ACP agents: {settings.acp_base_url}")
    print(f"  SearchMCP:  {settings.search_mcp_url}")
    print(f"  ReportMCP:  {settings.report_mcp_url}")
    print("-" * 50)
    session_thread = str(uuid.uuid4())

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            return

        await run_turn(user_input, session_thread)


if __name__ == "__main__":
    asyncio.run(main())
