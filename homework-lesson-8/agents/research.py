"""Research Agent: інструменти з hw5 без збереження звіту."""

from __future__ import annotations

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from config import RESEARCH_SYSTEM_PROMPT, Settings
from tools import knowledge_search, read_url, web_search


def build_research_agent():
    settings = Settings()
    model = ChatOpenAI(
        model=settings.model_name,
        api_key=settings.api_key.get_secret_value(),
        temperature=0,
    )
    return create_agent(
        model,
        tools=[knowledge_search, web_search, read_url],
        system_prompt=RESEARCH_SYSTEM_PROMPT,
    )
