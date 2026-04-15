"""Critic Agent: structured CritiqueResult + верифікаційні tools."""

from __future__ import annotations

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from config import CRITIC_SYSTEM_PROMPT, Settings
from schemas import CritiqueResult
from tools import knowledge_search, read_url, web_search


def build_critic_agent():
    settings = Settings()
    model = ChatOpenAI(
        model=settings.model_name,
        api_key=settings.api_key.get_secret_value(),
        temperature=0,
    )
    return create_agent(
        model,
        tools=[knowledge_search, web_search, read_url],
        system_prompt=CRITIC_SYSTEM_PROMPT,
        response_format=CritiqueResult,
    )
