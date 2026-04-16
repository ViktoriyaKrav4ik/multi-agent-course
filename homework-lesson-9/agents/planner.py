"""Planner Agent: structured ResearchPlan + web/knowledge tools (via MCP)."""

from __future__ import annotations

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from config import PLANNER_SYSTEM_PROMPT, Settings
from schemas import ResearchPlan

PLANNER_TOOL_NAMES = {"web_search", "knowledge_search"}


def build_planner_agent(mcp_tools: list):
    """Створити Planner agent з MCP-інструментами.

    Args:
        mcp_tools: LangChain tools, конвертовані з SearchMCP.
    """
    settings = Settings()
    model = ChatOpenAI(
        model=settings.model_name,
        api_key=settings.api_key.get_secret_value(),
        temperature=0,
    )
    tools = [t for t in mcp_tools if t.name in PLANNER_TOOL_NAMES]
    return create_agent(
        model,
        tools=tools,
        system_prompt=PLANNER_SYSTEM_PROMPT,
        response_format=ResearchPlan,
    )
