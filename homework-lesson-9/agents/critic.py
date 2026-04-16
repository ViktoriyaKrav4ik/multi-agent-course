"""Critic Agent: structured CritiqueResult + верифікаційні tools (via MCP)."""

from __future__ import annotations

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from config import CRITIC_SYSTEM_PROMPT, Settings
from schemas import CritiqueResult


def build_critic_agent(mcp_tools: list):
    """Створити Critic agent з MCP-інструментами.

    Args:
        mcp_tools: LangChain tools, конвертовані з SearchMCP.
    """
    settings = Settings()
    model = ChatOpenAI(
        model=settings.model_name,
        api_key=settings.api_key.get_secret_value(),
        temperature=0,
    )
    return create_agent(
        model,
        tools=mcp_tools,
        system_prompt=CRITIC_SYSTEM_PROMPT,
        response_format=CritiqueResult,
    )
