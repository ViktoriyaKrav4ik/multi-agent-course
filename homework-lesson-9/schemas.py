"""Pydantic-схеми для Planner і Critic (structured output)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ResearchPlan(BaseModel):
    goal: str = Field(description="Що саме потрібно з'ясувати або дослідити")
    search_queries: list[str] = Field(description="Конкретні пошукові запити для виконання")
    sources_to_check: list[str] = Field(
        description="Джерела: 'knowledge_base', 'web' або обидва (списком рядків)"
    )
    output_format: str = Field(description="Яким має бути фінальний звіт (структура, розділи)")


class CritiqueResult(BaseModel):
    verdict: Literal["APPROVE", "REVISE"]
    is_fresh: bool = Field(description="Чи дані актуальні, чи є свіжіші джерела")
    is_complete: bool = Field(description="Чи повністю покрито оригінальний запит користувача")
    is_well_structured: bool = Field(
        description="Чи логічно структуровані знахідки й готові стати звітом"
    )
    strengths: list[str] = Field(description="Що добре в дослідженні")
    gaps: list[str] = Field(description="Що відсутнє, застаріло або слабо структуровано")
    revision_requests: list[str] = Field(
        description="Конкретні вимоги до доопрацювання, якщо verdict=REVISE"
    )
