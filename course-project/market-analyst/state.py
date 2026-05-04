"""Стан LangGraph для циклу Analyst ↔ Critic та Compiler."""

from __future__ import annotations

from typing import TypedDict


class AgentState(TypedDict, total=False):
    topic: str
    scope: str
    focus_areas: list[str]
    session_id: str
    analyst_attempts: int
    draft: dict
    feedback: dict
    final_report: dict
    output_md_path: str
