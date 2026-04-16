"""Research Agent: всі інструменти SearchMCP (via MCP)."""

from __future__ import annotations

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from config import RESEARCH_SYSTEM_PROMPT, Settings


def build_research_agent(mcp_tools: list):
    """Створити Research agent з MCP-інструментами.

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
        system_prompt=RESEARCH_SYSTEM_PROMPT,
    )
