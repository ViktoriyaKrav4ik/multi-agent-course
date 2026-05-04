"""Research Analyst: RAG + web + structured DraftReport."""

from __future__ import annotations

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from config import Settings
from prompts import ANALYST_SYSTEM
from schemas import DraftReport
from tools import knowledge_search, read_url, web_search


def build_analyst_agent():
    s = Settings()
    model = ChatOpenAI(
        model=s.model_name,
        api_key=s.api_key.get_secret_value(),
        temperature=0,
    )
    return create_agent(
        model,
        tools=[knowledge_search, web_search, read_url],
        system_prompt=ANALYST_SYSTEM,
        response_format=DraftReport,
    )
