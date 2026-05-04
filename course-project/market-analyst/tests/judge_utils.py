"""Допоміжні функції для LLM-as-a-Judge тестів."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from config import Settings


class JudgeVerdict(BaseModel):
    """Результат оцінювання тесту суддею."""

    passed: bool = Field(description="Чи задовольняє артефакт критеріям")
    reasoning: str = Field(description="Коротке обґрунтування")


def llm_judge(*, criteria: str, artifact: str) -> JudgeVerdict:
    s = Settings()
    llm = ChatOpenAI(
        model=s.effective_judge_model(),
        api_key=s.api_key.get_secret_value(),
        temperature=0,
    )
    prompt = (
        "Ти — Judge для автоматизованих тестів мультиагентної системи. "
        "Оціни артефакт суворо за критеріями. Відповідь українською.\n\n"
        f"Критерії:\n{criteria}\n\nАртефакт для перевірки:\n{artifact}"
    )
    out = llm.with_structured_output(JudgeVerdict).invoke(
        [
            SystemMessage(content="Поверни структуровану оцінку JudgeVerdict."),
            HumanMessage(content=prompt),
        ]
    )
    return out if isinstance(out, JudgeVerdict) else JudgeVerdict.model_validate(out)
