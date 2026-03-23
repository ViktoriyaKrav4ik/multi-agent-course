"""Research Agent: LangGraph ReAct + пам'ять + RAG tool."""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from config import Settings, SYSTEM_PROMPT
from tools import knowledge_search, read_url, web_search, write_report

settings = Settings()

llm = ChatOpenAI(
    model=settings.model_name,
    api_key=settings.api_key.get_secret_value(),
    temperature=0,
)

tools = [knowledge_search, web_search, read_url, write_report]
memory = MemorySaver()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("placeholder", "{messages}"),
    ]
)

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=prompt,
    checkpointer=memory,
)
