"""LangGraph: Analyst → Critic з Command routing; Compiler → файл .md."""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from config import Settings
from prompts import COMPILER_SYSTEM
from schemas import CriticFeedback, DraftReport, FinalReport
from state import AgentState
from tools import compare_hybrid_years, knowledge_search, rank_corn_hybrids, web_search

_runtime_callbacks: list = []


def set_tracing_callbacks(handlers: list) -> None:
    global _runtime_callbacks
    _runtime_callbacks = list(handlers)


def _sub_cfg(state: AgentState, role: str, recursion_limit: int) -> dict:
    sid = state.get("session_id") or "session"
    iteration = int(state.get("analyst_attempts") or 0)
    cfg: dict = {
        "configurable": {"thread_id": f"{sid}-{role}-{uuid.uuid4().hex[:8]}"},
        "recursion_limit": recursion_limit,
        "metadata": {
            "agent": role,
            "session_id": sid,
            "analyst_critic_iteration": str(iteration),
        },
    }
    if _runtime_callbacks:
        cfg["callbacks"] = _runtime_callbacks
    return cfg


def _analyst_user_content(state: AgentState) -> str:
    lines = [
        f"Тема: {state.get('topic', '')}",
        f"Скоуп: {state.get('scope', '')}",
        f"Фокусні напрями: {', '.join(state.get('focus_areas') or [])}",
    ]
    fb = state.get("feedback")
    if fb:
        lines.append("")
        lines.append("Онови DraftReport з урахуванням feedback Критика:")
        lines.append(json.dumps(fb, ensure_ascii=False, indent=2))
    return "\n".join(lines)


def analyst_node(state: AgentState) -> dict:
    settings = Settings()
    # Керований збір контексту без agent-loop, щоб уникати recursion errors.
    topic = state.get("topic", "")
    scope = state.get("scope", "")
    focus = ", ".join(state.get("focus_areas") or [])
    kb_query = f"{topic}. {scope}. {focus}".strip()
    web_query = f"{topic} {scope} market trends 2024 2025 2026".strip()

    kb_ctx = knowledge_search.invoke({"query": kb_query})
    web_ctx = web_search.invoke({"query": web_query})
    ranking_ctx = ""
    ranking_obj: dict | None = None
    compare_ctx = ""
    compare_obj: dict | None = None
    topic_l = f"{topic} {scope} {focus}".lower()

    def _extract_hybrid_name(text: str) -> str | None:
        m = re.search(r"гібрид[ауі]?\s+([A-Za-zА-Яа-яІіЇїЄє0-9 ._-]+)", text, flags=re.IGNORECASE)
        if not m:
            return None
        name = m.group(1).strip()
        # stop at common separators
        name = re.split(r"[;,]|між|за|по", name, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        return name or None

    hybrid_name = _extract_hybrid_name(f"{topic} {scope}")
    is_compare = any(x in topic_l for x in ("порівня", "compare", "між 2024", "між 2025", "рік"))
    if is_compare and hybrid_name:
        compare_ctx = compare_hybrid_years.invoke({"hybrid_name": hybrid_name})
        try:
            parsed = json.loads(compare_ctx)
            if isinstance(parsed, dict) and isinstance(parsed.get("by_year"), list):
                compare_obj = parsed
        except Exception:
            compare_obj = None

    if compare_obj and compare_obj.get("by_year"):
        by_year = compare_obj["by_year"]
        by_loc = compare_obj.get("by_location_year", [])
        findings = []
        dps = []
        for row in by_year:
            yr = row.get("Year", row.get("Рік", "н/д"))
            y = row.get("yield_mean", "н/д")
            m = row.get("moisture_mean", "н/д")
            e = row.get("ebitda_mean", "н/д")
            n = row.get("n_obs", "н/д")
            findings.append(
                {
                    "title": f"{hybrid_name} — {yr}",
                    "detail": f"n_obs={n}; середня урожайність={round(float(y),3) if y!='н/д' else y}; "
                    f"середня вологість={round(float(m),3) if m!='н/д' else m}; "
                    f"середня EBITDA={round(float(e),3) if e!='н/д' else e}.",
                }
            )
            dps.append(f"{yr}: yield={y}, moisture={m}, EBITDA={e}, n_obs={n}")
        if by_loc:
            top_loc = by_loc[:10]
            for row in top_loc:
                dps.append(
                    f"loc-year {row.get('Year')}/{row.get('Village', row.get('Cluster_Village', 'loc'))}: "
                    f"yield={row.get('yield_mean')}, moisture={row.get('moisture_mean')}, EBITDA={row.get('ebitda_mean')}"
                )
        src = compare_obj.get("meta", {}).get("csv_source", "CSV corpus")
        draft = DraftReport(
            executive_summary=(
                f"Порівняння гібриду {hybrid_name} між роками побудовано детерміновано на CSV даних "
                "за показниками урожайності, вологості та EBITDA."
            ),
            findings=findings,
            sources=[str(src)],
            data_points=dps[:20],
        )
        prev = int(state.get("analyst_attempts") or 0)
        return {"draft": draft.model_dump(), "analyst_attempts": prev + 1}

    if any(x in topic_l for x in ("кукурудз", "corn", "гібрид", "hybrid")):
        ranking_ctx = rank_corn_hybrids.invoke(
            {"objective": "balanced", "top_n": 10, "max_moisture": 22.0}
        )
        try:
            parsed = json.loads(ranking_ctx)
            if isinstance(parsed, dict) and isinstance(parsed.get("ranking"), list):
                ranking_obj = parsed
        except Exception:
            ranking_obj = None

    # Якщо ranking з CSV валідний — будуємо драфт детерміновано без галюцинацій.
    if ranking_obj and ranking_obj.get("ranking"):
        ranking_items = ranking_obj["ranking"][:10]
        findings = []
        data_points = []
        for i, item in enumerate(ranking_items, 1):
            hybrid = item.get("hybrid", f"Гібрид {i}")
            y = item.get("yield_mean", "н/д")
            m = item.get("moisture_mean", "н/д")
            e = item.get("ebitda_mean", "н/д")
            n = item.get("n_obs", "н/д")
            score = item.get("score", "н/д")
            findings.append(
                {
                    "title": f"{i}. {hybrid}",
                    "detail": (
                        f"Середня урожайність: {y}; середня вологість: {m}; "
                        f"середня EBITDA: {e}; спостережень: {n}; score: {score}."
                    ),
                }
            )
            data_points.append(
                f"{hybrid}: yield={y}, moisture={m}, EBITDA={e}, n_obs={n}, score={score}"
            )

        src = ranking_obj.get("meta", {}).get("csv_source", "CSV corpus")
        draft = DraftReport(
            executive_summary=(
                "Рейтинг сформовано детерміновано на основі CSV з дослідних полів "
                "(2024-2025) за метриками урожайності, вологості, EBITDA та стабільності."
            ),
            findings=findings,
            sources=[str(src)],
            data_points=data_points,
        )
        prev = int(state.get("analyst_attempts") or 0)
        return {"draft": draft.model_dump(), "analyst_attempts": prev + 1}

    llm = ChatOpenAI(
        model=settings.model_name,
        api_key=settings.api_key.get_secret_value(),
        temperature=0,
    )
    prompt = (
        "Ти Research Analyst. Сформуй DraftReport за схемою строго на основі контексту.\n"
        "Вимоги: 5-10 sources, конкретні findings, збалансований виклад.\n\n"
        f"Тема: {topic}\nСкоуп: {scope}\nФокус: {focus}\n\n"
        f"Локальний контекст (RAG):\n{kb_ctx}\n\n"
        f"Web-контекст:\n{web_ctx}\n\n"
        f"Аналітичний ranking з CSV (детермінований):\n{ranking_ctx}\n\n"
        "Якщо ranking з CSV надано, будуй ТОП гібридів насамперед на ньому.\n"
        "PDF/RAG використовуй як пояснення і контекст.\n"
        "Якщо інформації бракує, явно вкажи обмеження в findings/detail."
    )
    structured = llm.with_structured_output(DraftReport).invoke(
        [HumanMessage(content=prompt)],
        config=_sub_cfg(state, "analyst", settings.max_analyst_steps),
    )
    draft = structured if isinstance(structured, DraftReport) else DraftReport.model_validate(structured)
    prev = int(state.get("analyst_attempts") or 0)
    return {
        "draft": draft.model_dump(),
        "analyst_attempts": prev + 1,
    }


def critic_node(state: AgentState) -> Command[Literal["analyst", "compiler"]]:
    settings = Settings()
    draft = state.get("draft")
    if not draft:
        raise RuntimeError("Critic: порожній draft")

    draft_model = DraftReport.model_validate(draft)
    fact_ctx = web_search.invoke({"query": f"{state.get('topic', '')} latest market data risks"})
    llm = ChatOpenAI(
        model=settings.model_name,
        api_key=settings.api_key.get_secret_value(),
        temperature=0,
    )
    crit_prompt = (
        "Ти Critic (devil's advocate). Оціни DraftReport та поверни CriticFeedback.\n"
        "Шукай упередженість, логічні прогалини, необґрунтовані твердження.\n\n"
        f"DraftReport:\n{draft_model.model_dump_json(indent=2, ensure_ascii=False)}\n\n"
        f"Додатковий контекст для fact-check:\n{fact_ctx}"
    )
    structured = llm.with_structured_output(CriticFeedback).invoke(
        [HumanMessage(content=crit_prompt)],
        config=_sub_cfg(state, "critic", settings.max_critic_steps),
    )
    fb = structured if isinstance(structured, CriticFeedback) else CriticFeedback.model_validate(structured)
    fb_dict = fb.model_dump()
    attempts = int(state.get("analyst_attempts") or 0)
    max_i = settings.max_analyst_critic_iterations
    if fb.verdict == "NEEDS_REVISION" and attempts < max_i:
        return Command(goto="analyst", update={"feedback": fb_dict})
    return Command(goto="compiler", update={"feedback": fb_dict})


def _final_to_markdown(fr: FinalReport) -> str:
    return "\n".join(
        [
            "# Звіт: ринкове дослідження",
            "",
            "## Executive summary",
            fr.executive_summary,
            "",
            "## Ключові знахідки",
            *[f"- {x}" for x in fr.key_findings],
            "",
            "## Рекомендації",
            *[f"- {x}" for x in fr.recommendations],
            "",
            "## Джерела",
            *[f"- {x}" for x in fr.sources],
            "",
            "## Методологія",
            fr.methodology,
        ]
    )


def compiler_node(state: AgentState) -> dict:
    settings = Settings()
    draft = DraftReport.model_validate(state["draft"])
    fb = state.get("feedback")
    meta_note = ""
    if isinstance(fb, dict) and fb.get("verdict") == "NEEDS_REVISION":
        meta_note = (
            "\nУвага: verdict NEEDS_REVISION, але досягнуто ліміт ітерацій Analyst↔Critic — "
            "компілюємо останній чорновик."
        )

    human = (
        f"Тема: {state.get('topic')}\nСкоуп: {state.get('scope')}\n\n"
        f"DraftReport:\n{draft.model_dump_json(indent=2, ensure_ascii=False)}\n\n"
        f"Feedback критика:\n{json.dumps(fb, ensure_ascii=False, indent=2) if fb else 'немає'}"
        f"{meta_note}"
    )

    llm = ChatOpenAI(
        model=settings.model_name,
        api_key=settings.api_key.get_secret_value(),
        temperature=0,
    )
    structured = llm.with_structured_output(FinalReport).invoke(
        [
            {"role": "system", "content": COMPILER_SYSTEM},
            HumanMessage(content=human),
        ],
        config=_sub_cfg(state, "compiler", 16),
    )
    final = (
        structured
        if isinstance(structured, FinalReport)
        else FinalReport.model_validate(structured)
    )

    out_dir = Path(__file__).resolve().parent / settings.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sid = (state.get("session_id") or "run")[:12]
    name = f"report_{sid}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    path = out_dir / name
    path.write_text(_final_to_markdown(final), encoding="utf-8")

    return {
        "final_report": final.model_dump(),
        "output_md_path": str(path.resolve()),
    }


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("analyst", analyst_node)
    g.add_node("critic", critic_node)
    g.add_node("compiler", compiler_node)
    g.add_edge(START, "analyst")
    g.add_edge("analyst", "critic")
    g.add_edge("compiler", END)
    return g.compile()


app = build_graph()
