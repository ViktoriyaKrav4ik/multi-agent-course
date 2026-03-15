"""Research Agent: LLM, tools, пам'ять, ReAct-цикл через LangGraph."""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from config import Settings, SYSTEM_PROMPT
from tools import web_search, read_url, write_report

settings = Settings()

llm = ChatOpenAI(
    model=settings.model_name,
    api_key=settings.api_key.get_secret_value(),
    temperature=0,
)

tools = [web_search, read_url, write_report]
memory = MemorySaver()

# Новий API: замість state_modifier використовується prompt (ChatPromptTemplate)
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("placeholder", "{messages}"),
])

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=prompt,
    checkpointer=memory,
)
