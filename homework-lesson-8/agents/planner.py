"""Planner Agent: structured ResearchPlan + web/knowledge tools."""

from __future__ import annotations

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from config import PLANNER_SYSTEM_PROMPT, Settings
from schemas import ResearchPlan
from tools import knowledge_search, web_search


def build_planner_agent():
    settings = Settings()
    model = ChatOpenAI(
        model=settings.model_name,
        api_key=settings.api_key.get_secret_value(),
        temperature=0,
    )
    return create_agent(
        model,
        tools=[web_search, knowledge_search],
        system_prompt=PLANNER_SYSTEM_PROMPT,
        response_format=ResearchPlan,
    )
