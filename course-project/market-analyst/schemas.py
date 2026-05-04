"""Pydantic контракти з ТЗ course-project/project_market_analyst.md."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Finding(BaseModel):
    title: str = Field(description="Короткий заголовок знахідки")
    detail: str = Field(description="Опис з фактами та посиланнями на джерела в тексті")


class DraftReport(BaseModel):
    executive_summary: str
    findings: list[Finding]
    sources: list[str]
    data_points: list[str]


class CriticFeedback(BaseModel):
    verdict: Literal["APPROVED", "NEEDS_REVISION"]
    issues: list[str]
    missing_perspectives: list[str]
    fact_check_results: list[str]
    score: float = Field(ge=0.0, le=1.0, description="Якість чорновика 0–1")


class FinalReport(BaseModel):
    executive_summary: str
    key_findings: list[str]
    recommendations: list[str]
    sources: list[str]
    methodology: str
