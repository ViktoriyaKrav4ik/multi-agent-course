import os
import tempfile
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore[misc, assignment]

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT = Path(__file__).resolve().parent
_ENV_FILE = _ROOT / ".env"

if load_dotenv is not None:
    load_dotenv(_ENV_FILE, encoding="utf-8")


class Settings(BaseSettings):
    api_key: SecretStr = Field(alias="OPENAI_API_KEY")
    model_name: str = Field(alias="MODEL_NAME", default="gpt-4o-mini")
    judge_model_name: str = Field(default="", alias="JUDGE_MODEL_NAME")

    max_search_results: int = 5
    max_url_content_length: int = 5000
    max_web_search_length: int = 8000

    embedding_model: str = "text-embedding-3-small"
    data_dir: str = "corpus"
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
    max_analyst_critic_iterations: int = 5
    max_analyst_steps: int = 80
    max_critic_steps: int = 40

    langfuse_public_key: str = Field(default="", alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field(default="", alias="LANGFUSE_SECRET_KEY")
    langfuse_base_url: str = Field(
        default="https://us.cloud.langfuse.com", alias="LANGFUSE_BASE_URL"
    )

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

    @field_validator("langfuse_public_key", "langfuse_secret_key", mode="before")
    @classmethod
    def _strip_langfuse_keys(cls, v: object) -> str:
        if v is None:
            return ""
        return str(v).strip()

    @field_validator("langfuse_base_url", mode="before")
    @classmethod
    def _strip_langfuse_url(cls, v: object) -> str:
        s = str(v or "").strip().rstrip("/")
        return s or "https://us.cloud.langfuse.com"

    def langfuse_configured(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    def effective_judge_model(self) -> str:
        j = (self.judge_model_name or "").strip()
        return j or self.model_name


def apply_langfuse_env_from_settings() -> None:
    try:
        s = Settings()
    except Exception:
        return
    if s.langfuse_public_key:
        os.environ["LANGFUSE_PUBLIC_KEY"] = s.langfuse_public_key.strip()
    if s.langfuse_secret_key:
        os.environ["LANGFUSE_SECRET_KEY"] = s.langfuse_secret_key.strip()
    if s.langfuse_base_url:
        base = s.langfuse_base_url.strip().rstrip("/")
        os.environ["LANGFUSE_BASE_URL"] = base
        os.environ["LANGFUSE_HOST"] = base


apply_langfuse_env_from_settings()


def get_rag_index_path(settings: Settings) -> Path:
    override = (settings.rag_index_dir or os.environ.get("RAG_INDEX_DIR", "") or "").strip()
    if override:
        return Path(override).expanduser().resolve()

    candidate = (_ROOT / settings.index_dir).resolve()
    try:
        str(candidate).encode("ascii")
    except UnicodeEncodeError:
        base = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
        return (Path(base) / "multi-agent-course-market-analyst-faiss").resolve()
    return candidate
