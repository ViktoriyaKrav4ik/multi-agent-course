import os
import tempfile
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT = Path(__file__).resolve().parent
_ENV_FILE = _ROOT / ".env"


class Settings(BaseSettings):
    """Налаштування з .env: OPENAI_API_KEY, опційно MODEL_NAME."""

    api_key: SecretStr = Field(alias="OPENAI_API_KEY")
    model_name: str = Field(alias="MODEL_NAME", default="gpt-4o-mini")

    # Web search
    max_search_results: int = 5
    max_url_content_length: int = 5000
    max_web_search_length: int = 8000

    # RAG
    embedding_model: str = "text-embedding-3-small"
    data_dir: str = "data"
    index_dir: str = "index"
    # Опційно: власний ASCII-шлях для FAISS (Windows + кирилиця в шляху проєкту)
    rag_index_dir: str | None = Field(default=None, alias="RAG_INDEX_DIR")
    chunk_size: int = 500
    chunk_overlap: int = 100
    retrieval_top_k: int = 10
    rerank_top_n: int = 3
    rerank_model: str = "BAAI/bge-reranker-base"
    ensemble_weights: tuple[float, float] = (0.5, 0.5)
    # false — без HuggingFace reranker; зазвичай стабільніші метадані (сторінка PDF) і швидше
    use_reranking: bool = Field(default=True, alias="USE_RERANKING")

    # Agent
    output_dir: str = "output"
    max_iterations: int = 10

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


def get_rag_index_path(settings: Settings) -> Path:
    """
    Каталог для FAISS + chunks.pkl.

    На Windows нативний FAISS (FileIOWriter) часто не може створити файли, якщо у повному
    шляху є не-ASCII (наприклад кирилиця в «Документи») — тоді повертаємо ASCII-шлях
    у %LOCALAPPDATA%. Можна перевизначити: змінна середовища RAG_INDEX_DIR.
    """
    override = (settings.rag_index_dir or os.environ.get("RAG_INDEX_DIR", "") or "").strip()
    if override:
        return Path(override).expanduser().resolve()

    candidate = (_ROOT / settings.index_dir).resolve()
    try:
        str(candidate).encode("ascii")
    except UnicodeEncodeError:
        base = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
        return (Path(base) / "multi-agent-course-hw5-faiss-index").resolve()
    return candidate


SYSTEM_PROMPT = """Ти — Research Agent з локальною базою знань (RAG) і доступом до інтернету.

Доступні інструменти:
1. knowledge_search(query) — пошук у завантажених документах (PDF/текст з папки data/). Використовуй для питань про зміст цих матеріалів, визначення термінів з файлів, цитат зі сторінок документів.
2. web_search(query) — пошук в інтернеті (заголовки, URL, сніпети).
3. read_url(url) — повний текст сторінки за посиланням.
4. write_report(filename, content) — зберегти Markdown-звіт у файл (output/). filename — латиниця, цифри, підкреслення, розширення .md.

Стратегія:
- Якщо питання стосується тем із локальних матеріалів або формулювання схоже на «що сказано в документах / у статті / у PDF» — спочатку knowledge_search.
- Для актуальних новин, порівнянь з 2024–2026, зовнішніх джерел — web_search і за потреби read_url.
- Можна комбінувати: локальна база + веб для повноти.
- Після knowledge_search у результаті інструменту є блок, що починається з «ДЛЯ ВІДПОВІДІ КОРИСТУВАЧУ». **Обов'язково** скопіюй розділ **«Джерела (PDF):»** з цього блоку **дослівно** у свою відповідь користувачу (можна одразу після основного тексту). Так само включи цей список у **write_report**.
- У **кожній** відповіді та у **write_report** для фактів з PDF вказуй **файл і номер сторінки** (як у тому блоці). Не подавай цитати з PDF без сторінки.
- Для веб-джерел у звіті вказуй URL.
- Обов'язково виклич write_report з повним structured Markdown, потім повідом шлях, який повернув інструмент. Не вигадуй шлях до файлу.
- Відповідай українською, якщо користувач пише українською.
"""
