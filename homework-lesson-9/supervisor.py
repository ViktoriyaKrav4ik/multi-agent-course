"""
Supervisor: оркестратор plan → research → critique → save_report.

- Делегує агентам через ACP (acp_sdk.client.Client)
- Зберігає звіти через ReportMCP (fastmcp.Client)
- HITL на save_report через HumanInTheLoopMiddleware
"""

from __future__ import annotations

import asyncio

from acp_sdk.client import Client as ACPClient
from acp_sdk.models import Message, MessagePart
from fastmcp import Client as MCPClient
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from config import SUPERVISOR_SYSTEM_PROMPT, Settings

settings = Settings()

ACP_BASE = settings.acp_base_url
ACP_HEADERS = {"Content-Type": "application/json"}
REPORT_MCP_URL = settings.report_mcp_url


# ── ACP delegation tools ───────────────────────────────────────

@tool
async def delegate_to_planner(request: str) -> str:
    """Побудувати структурований план дослідження (goal, search_queries, sources, output_format) через ACP Planner."""
    async with ACPClient(base_url=ACP_BASE, headers=ACP_HEADERS) as client:
        run = await client.run_sync(
            agent="planner",
            input=[Message(role="user", parts=[MessagePart(content=request)])],
        )
    return run.output[-1].parts[0].content


@tool
async def delegate_to_researcher(request: str) -> str:
    """Виконати дослідження за планом або інструкціями (knowledge_search, web_search, read_url) через ACP Researcher."""
    async with ACPClient(base_url=ACP_BASE, headers=ACP_HEADERS) as client:
        run = await client.run_sync(
            agent="researcher",
            input=[Message(role="user", parts=[MessagePart(content=request)])],
        )
    return run.output[-1].parts[0].content


@tool
async def delegate_to_critic(findings: str) -> str:
    """Оцінити якість дослідження через ACP Critic; повертає verdict: APPROVE або REVISE."""
    async with ACPClient(base_url=ACP_BASE, headers=ACP_HEADERS) as client:
        run = await client.run_sync(
            agent="critic",
            input=[Message(role="user", parts=[MessagePart(content=findings)])],
        )
    return run.output[-1].parts[0].content


# ── save_report via ReportMCP ──────────────────────────────────

@tool
async def save_report(filename: str, content: str) -> str:
    """Зберегти Markdown-звіт у output/ через ReportMCP. Потребує підтвердження користувача (HITL)."""
    async with MCPClient(REPORT_MCP_URL) as mcp_client:
        result = await mcp_client.call_tool(
            "save_report", {"filename": filename, "content": content}
        )
    return str(result)


# ── Build Supervisor ───────────────────────────────────────────

def build_supervisor():
    s = Settings()
    llm = ChatOpenAI(
        model=s.model_name,
        api_key=s.api_key.get_secret_value(),
        temperature=0,
    )
    prompt = SUPERVISOR_SYSTEM_PROMPT.format(max_revision_rounds=s.max_revision_rounds)
    return create_agent(
        llm,
        tools=[delegate_to_planner, delegate_to_researcher, delegate_to_critic, save_report],
        system_prompt=prompt,
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={"save_report": True},
                description_prefix="⏸️  Збереження звіту потребує підтвердження",
            ),
        ],
        checkpointer=InMemorySaver(),
    )


supervisor = build_supervisor()
