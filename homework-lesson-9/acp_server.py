"""
ACP Server: три агенти (planner, researcher, critic) на одному endpoint.

Порт: 8903 (за замовчуванням).
Кожен агент підключається до SearchMCP через fastmcp.Client і
використовує LangChain create_agent.

Запуск: python acp_server.py
"""

from __future__ import annotations

from acp_sdk.models import Message, MessagePart
from acp_sdk.server import Server
from fastmcp import Client as MCPClient
from langchain_core.messages import AIMessage

from agents.critic import build_critic_agent
from agents.planner import build_planner_agent
from agents.research import build_research_agent
from config import Settings
from mcp_utils import mcp_tools_to_langchain

settings = Settings()
server = Server()

SEARCH_MCP_URL = settings.search_mcp_url


def _last_assistant_text(result: dict) -> str:
    """Витягти текст останнього повідомлення агента."""
    msgs = result.get("messages") or []
    if not msgs:
        return ""
    last = msgs[-1]
    if isinstance(last, AIMessage) and last.content:
        return last.content if isinstance(last.content, str) else str(last.content)
    text = getattr(last, "text", None)
    if text:
        return str(text)
    return str(getattr(last, "content", "") or "")


# ── Planner Agent ───────────────────────────────────────────────

@server.agent(
    name="planner",
    description="Складає структурований план дослідження (ResearchPlan). "
                "Використовує web_search і knowledge_search через MCP.",
)
async def planner_handler(input: list[Message]) -> Message:
    user_text = input[-1].parts[0].content

    async with MCPClient(SEARCH_MCP_URL) as mcp_client:
        mcp_tools = await mcp_client.list_tools()
        lc_tools = mcp_tools_to_langchain(mcp_tools, mcp_client)

        agent = build_planner_agent(lc_tools)
        result = await agent.ainvoke({"messages": [("user", user_text)]})

    structured = result.get("structured_response")
    if structured is not None:
        content = structured.model_dump_json(indent=2, ensure_ascii=False)
    else:
        content = _last_assistant_text(result)

    return Message(role="agent", parts=[MessagePart(content=content)])


# ── Research Agent ──────────────────────────────────────────────

@server.agent(
    name="researcher",
    description="Виконує дослідження за планом: knowledge_search, web_search, read_url через MCP.",
)
async def researcher_handler(input: list[Message]) -> Message:
    user_text = input[-1].parts[0].content

    async with MCPClient(SEARCH_MCP_URL) as mcp_client:
        mcp_tools = await mcp_client.list_tools()
        lc_tools = mcp_tools_to_langchain(mcp_tools, mcp_client)

        agent = build_research_agent(lc_tools)
        result = await agent.ainvoke({"messages": [("user", user_text)]})

    content = _last_assistant_text(result)
    return Message(role="agent", parts=[MessagePart(content=content)])


# ── Critic Agent ────────────────────────────────────────────────

@server.agent(
    name="critic",
    description="Перевіряє якість дослідження. Повертає CritiqueResult (verdict: APPROVE/REVISE).",
)
async def critic_handler(input: list[Message]) -> Message:
    user_text = input[-1].parts[0].content

    async with MCPClient(SEARCH_MCP_URL) as mcp_client:
        mcp_tools = await mcp_client.list_tools()
        lc_tools = mcp_tools_to_langchain(mcp_tools, mcp_client)

        agent = build_critic_agent(lc_tools)
        result = await agent.ainvoke({"messages": [("user", user_text)]})

    structured = result.get("structured_response")
    if structured is not None:
        content = structured.model_dump_json(indent=2, ensure_ascii=False)
    else:
        content = _last_assistant_text(result)

    return Message(role="agent", parts=[MessagePart(content=content)])


# ── Entrypoint ──────────────────────────────────────────────────

if __name__ == "__main__":
    port = settings.acp_port
    print(f"🤖 ACP Server starting on port {port} (planner, researcher, critic)...")
    print(f"   SearchMCP: {SEARCH_MCP_URL}")
    server.run(port=port)
