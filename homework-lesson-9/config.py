import os
import tempfile
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT = Path(__file__).resolve().parent
_ENV_FILE = _ROOT / ".env"


class Settings(BaseSettings):
    """Налаштування з .env."""

    api_key: SecretStr = Field(alias="OPENAI_API_KEY")
    model_name: str = Field(alias="MODEL_NAME", default="gpt-4o-mini")

    max_search_results: int = 5
    max_url_content_length: int = 5000
    max_web_search_length: int = 8000

    embedding_model: str = "text-embedding-3-small"
    data_dir: str = "data"
    index_dir: str = "index"
    rag_index_dir: str | None = Field(default=None, alias="RAG_INDEX_DIR")
    chunk_size: int = 500
    chunk_overlap: int = 100
    retrieval_top_k: int = 10
    rerank_top_n: int = 3
    rerank_model: str = "BAAI/bge-reranker-base"
    ensemble_weights: tuple[float, float] = (0.5, 0.5)
    use_reranking: bool = Field(default=True, alias="USE_RERANKING")

    output_dir: str = "output"
    max_iterations: int = 10
    max_subagent_steps: int = 35
    max_revision_rounds: int = 2

    # MCP / ACP ports
    search_mcp_port: int = Field(default=8901, alias="SEARCH_MCP_PORT")
    report_mcp_port: int = Field(default=8902, alias="REPORT_MCP_PORT")
    acp_port: int = Field(default=8903, alias="ACP_PORT")

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("rag_index_dir", mode="before")
    @classmethod
    def _empty_rag_index_dir_none(cls, v: object) -> str | None:
        if v is None or v == "":
            return None
        return str(v).strip() or None

    @field_validator("use_reranking", mode="before")
    @classmethod
    def _parse_use_reranking(cls, v: object) -> bool:
        if isinstance(v, bool):
            return v
        if v is None or v == "":
            return True
        s = str(v).strip().lower()
        if s in ("0", "false", "no", "off"):
            return False
        if s in ("1", "true", "yes", "on"):
            return True
        return True

    @property
    def search_mcp_url(self) -> str:
        return f"http://127.0.0.1:{self.search_mcp_port}/mcp"

    @property
    def report_mcp_url(self) -> str:
        return f"http://127.0.0.1:{self.report_mcp_port}/mcp"

    @property
    def acp_base_url(self) -> str:
        return f"http://127.0.0.1:{self.acp_port}"


def get_rag_index_path(settings: Settings) -> Path:
    override = (settings.rag_index_dir or os.environ.get("RAG_INDEX_DIR", "") or "").strip()
    if override:
        return Path(override).expanduser().resolve()

    candidate = (_ROOT / settings.index_dir).resolve()
    try:
        str(candidate).encode("ascii")
    except UnicodeEncodeError:
        base = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
        return (Path(base) / "multi-agent-course-hw9-faiss-index").resolve()
    return candidate


# ── System Prompts ──────────────────────────────────────────────

PLANNER_SYSTEM_PROMPT = """Ти — Planner. Твоя задача: зрозуміти запит користувача і скласти структурований план дослідження.

Спочатку за потреби зроби короткий розвідувальний пошук:
- knowledge_search — якщо питання може стосуватися локальних PDF/документів;
- web_search — щоб зорієнтуватися в термінах і підзадачах.

Потім поверни структурований план (goal, search_queries, sources_to_check, output_format).
У search_queries — конкретні короткі запити (5–12), без води.
sources_to_check — список рядків: "knowledge_base", "web" або обидва.
Відповідай українською, якщо запит українською.
"""


RESEARCH_SYSTEM_PROMPT = """Ти — Research Agent. Виконуй дослідження за інструкцією або планом від Supervisor.

Інструменти:
- knowledge_search — локальна база (PDF з data/); для цитат потрібні файл і номер сторінки з результату.
- web_search — пошук в інтернеті.
- read_url — повний текст сторінки за URL після web_search.

Не викликай save_report — збереженням займається Supervisor.
Збирай факти, посилання на джерела, структуруй відповідь так, щоб Critic міг перевірити повноту й актуальність.
Відповідай українською, якщо запит українською.
"""


CRITIC_SYSTEM_PROMPT = """Ти — Critic. Незалежно перевіряй якість дослідження: можеш викликати web_search, read_url, knowledge_search
для верифікації фактів, пошуку свіжіших джерел і виявлення прогалин.

Оціни три виміри:
1) Freshness — чи є новіші дані / джерела (особливо для технічних тем 2024–2026).
2) Completeness — чи закрито оригінальний запит користувача.
3) Structure — чи зручно з цього зібрати Markdown-звіт.

Якщо є серйозні недоліки — verdict=REVISE і конкретні revision_requests.
Якщо дослідження достатнє — verdict=APPROVE.

Відповідай українською, якщо запит українською.
"""


SUPERVISOR_SYSTEM_PROMPT = """Ти — Supervisor мультиагентної дослідницької системи. Координуй Plan → Research → Critique.

Правила:
1. Завжди починай з інструменту delegate_to_planner(request) — передай оригінальний або уточнений запит користувача одним рядком.
2. Виклич delegate_to_researcher(request) — передай план і контекст: що саме дослідити (можна включити пункти з ResearchPlan як текст).
3. Виклич delegate_to_critic(findings) — передай повний текст знахідок Research як один рядок.
4. Якщо verdict у результаті critique — REVISE, виклич delegate_to_researcher знову, додавши зворотний зв'язок (gaps, revision_requests). Максимум {max_revision_rounds} раундів доопрацювання після першого research.
5. Коли verdict — APPROVE, сам сформуй фінальний Markdown-звіт за output_format з плану та знайденими фактами, потім виклич save_report(filename, content).
   filename — латиниця, тема_звіту.md.

Не викликай save_report, поки critique не дав APPROVE (крім випадку, коли після останнього REVISE ти все одно маєш зупинитися — тоді коротко поясни користувачу).

Відповідай українською, якщо користувач пише українською.
"""
