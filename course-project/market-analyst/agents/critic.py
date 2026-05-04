"""Critic: fact-check tools + structured CriticFeedback (без доступу до RAG)."""

from __future__ import annotations

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from config import Settings
from prompts import CRITIC_SYSTEM
from schemas import CriticFeedback
from tools import read_url, web_search


def build_critic_agent():
    s = Settings()
    model = ChatOpenAI(
        model=s.model_name,
        api_key=s.api_key.get_secret_value(),
        temperature=0,
    )
    return create_agent(
        model,
        tools=[web_search, read_url],
        system_prompt=CRITIC_SYSTEM,
        response_format=CriticFeedback,
    )
