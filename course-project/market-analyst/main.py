"""
CLI для пайплайну «Аналітик ринку» + Langfuse callbacks на всіх викликах.

Запуск з каталогу market-analyst:
  python main.py --topic "..." --scope "..." --focus "a,b,c"
"""

from __future__ import annotations

import argparse
import logging
import uuid

from config import Settings
from graph import app, set_tracing_callbacks
from state import AgentState

logger = logging.getLogger(__name__)
settings = Settings()

_langfuse_handler = None
LANGFUSE_OK = False
if settings.langfuse_configured():
    try:
        from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler

        _langfuse_handler = LangfuseCallbackHandler()
        LANGFUSE_OK = True
    except Exception as e:
        logger.warning("Langfuse CallbackHandler недоступний: %s", e)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Market Analyst multi-agent pipeline")
    p.add_argument(
        "--topic",
        default="Агроринок України: зерно та олійні",
        help="Тема дослідження",
    )
    p.add_argument(
        "--scope",
        default="Ціни, логістика, добрива/ЗЗР, технології 2024–2026",
        help="Межі скоупу",
    )
    p.add_argument(
        "--focus",
        default="експорт,вартість інпутів,precision ag",
        help="Через кому: focus_areas",
    )
    return p.parse_args()


def _build_initial_state(args: argparse.Namespace) -> AgentState:
    focus = [x.strip() for x in args.focus.split(",") if x.strip()]
    session_id = str(uuid.uuid4())
    return {
        "topic": args.topic,
        "scope": args.scope,
        "focus_areas": focus,
        "session_id": session_id,
        "analyst_attempts": 0,
    }


def run_pipeline(state: AgentState) -> AgentState:
    handlers = [h for h in [_langfuse_handler] if h is not None]
    set_tracing_callbacks(handlers)

    cfg: dict = {
        "recursion_limit": settings.max_analyst_critic_iterations * 8 + 24,
        "metadata": {
            "agent": "graph",
            "session_id": state.get("session_id", ""),
            "analyst_critic_iteration": "0",
        },
    }
    if handlers:
        cfg["callbacks"] = handlers

    result = app.invoke(state, config=cfg)
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    state = _build_initial_state(args)
    print(f"session_id={state['session_id']}")
    if LANGFUSE_OK:
        print("Langfuse: tracing увімкнено")
    else:
        print("Langfuse: ключі не задані — тільки консоль")

    out = run_pipeline(state)
    path = out.get("output_md_path", "")
    print(f"Готово. Звіт: {path}")
    if out.get("final_report"):
        fr = out["final_report"]
        if isinstance(fr, dict) and fr.get("executive_summary"):
            print("\n--- Executive summary (приклад) ---\n")
            print(fr["executive_summary"][:1200])


if __name__ == "__main__":
    main()
