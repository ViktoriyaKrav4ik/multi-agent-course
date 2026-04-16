"""
SearchMCP Server — web_search, read_url, knowledge_search.

Порт: 8901 (за замовчуванням).
Використовується трьома ACP-агентами одночасно.

Запуск: python mcp_servers/search_mcp.py
"""

from __future__ import annotations

import os
import sys
from contextlib import redirect_stderr
from datetime import datetime
from io import StringIO
from pathlib import Path

# Додаємо батьківську директорію в sys.path для імпорту config / retriever
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastmcp import FastMCP

from config import Settings

settings = Settings()
mcp = FastMCP(name="SearchMCP")


# ── Resources ───────────────────────────────────────────────────

@mcp.resource("resource://knowledge-base-stats")
def knowledge_base_stats() -> str:
    """Кількість документів у knowledge base та дата останнього оновлення індексу."""
    import json

    from config import get_rag_index_path

    index_path = get_rag_index_path(settings)
    faiss_file = index_path / "index.faiss"
    chunks_file = index_path / "chunks.pkl"

    if not faiss_file.exists():
        return json.dumps({"status": "not_indexed", "documents": 0}, ensure_ascii=False)

    import pickle

    num_chunks = 0
    if chunks_file.exists():
        with open(chunks_file, "rb") as f:
            chunks = pickle.load(f)
        num_chunks = len(chunks)

    last_modified = datetime.fromtimestamp(faiss_file.stat().st_mtime).isoformat()
    return json.dumps(
        {
            "status": "ready",
            "num_chunks": num_chunks,
            "last_updated": last_modified,
            "index_path": str(index_path),
        },
        ensure_ascii=False,
    )


# ── Tools ───────────────────────────────────────────────────────

@mcp.tool()
def web_search(query: str) -> str:
    """Пошук в інтернеті за запитом. Повертає заголовки, URL та короткі сніпети."""
    try:
        from ddgs import DDGS
    except ImportError:
        return "Помилка: встановіть ddgs: pip install ddgs"

    try:
        with redirect_stderr(StringIO()):
            results = DDGS().text(query, max_results=settings.max_search_results)
    except Exception as e:
        return f"Помилка пошуку: {e}"

    if not results:
        return "Результатів не знайдено."

    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        href = r.get("href", "")
        body = (r.get("body") or "").strip()
        lines.append(f"{i}. {title}\n   URL: {href}\n   {body}")
    text = "\n\n".join(lines)
    max_len = settings.max_web_search_length
    if len(text) > max_len:
        text = text[:max_len] + "\n\n[... результати пошуку обрізано ...]"
    return text


@mcp.tool()
def read_url(url: str) -> str:
    """Отримати основний текст сторінки за URL."""
    try:
        import trafilatura
    except ImportError:
        return "Помилка: встановіть trafilatura: pip install trafilatura"

    if not url.startswith(("http://", "https://")):
        return "Помилка: URL має починатися з http:// або https://."

    try:
        downloaded = trafilatura.fetch_url(url)
    except Exception as e:
        return f"Помилка завантаження URL: {e}"

    if not downloaded:
        return "Не вдалося завантажити сторінку."

    text = trafilatura.extract(downloaded)
    if not text or not text.strip():
        return "Не вдалося витягнути текст із сторінки."

    max_len = settings.max_url_content_length
    if len(text) > max_len:
        text = text[:max_len] + "\n\n[... текст обрізано ...]"
    return text


@mcp.tool()
def knowledge_search(query: str) -> str:
    """Пошук у локальній базі знань (індексовані PDF/текст з data/). Hybrid + reranking."""
    from retriever import hybrid_search, index_ready

    if not index_ready():
        return (
            "Локальний індекс ще не створено. З каталогу homework-lesson-9 виконайте: "
            "python ingest.py (попередньо покладіть файли в data/)."
        )

    try:
        docs = hybrid_search(query)
    except Exception as e:
        return f"Помилка knowledge_search: {e}"

    if not docs:
        return "За цим запитом у локальній базі нічого релевантного не знайдено."

    lines = [
        "Кожен блок нижче має **Сторінка PDF** (з 1) та **Файл**.\n",
        f"Знайдено фрагментів: {len(docs)}\n",
    ]
    citation_lines: list[str] = []

    for i, doc in enumerate(docs, 1):
        meta = doc.metadata or {}
        src = meta.get("source") or meta.get("file_name") or meta.get("file_path", "невідомо")
        if isinstance(src, str) and len(src) > 80:
            src = os.path.basename(src)

        page = meta.get("page")
        if page is not None:
            try:
                page_human = int(page) + 1
                page_line = f"Сторінка PDF: **{page_human}**"
                cite_page = str(page_human)
            except (TypeError, ValueError):
                page_line = f"Сторінка PDF: **{page}**"
                cite_page = str(page)
        else:
            page_line = "Сторінка PDF: **н/д** (немає номера в метаданих)"
            cite_page = "н/д"

        citation_lines.append(f"- {src}, стор. {cite_page}")

        snippet = (doc.page_content or "").strip()
        if len(snippet) > 1200:
            snippet = snippet[:1200] + "\n[...]"
        lines.append(
            f"---\n### Фрагмент {i}\n"
            f"- {page_line}\n"
            f"- Файл: **{src}**\n\n"
            f"Текст:\n{snippet}"
        )

    lines.append("")
    lines.append("<<< ДЛЯ ВІДПОВІДІ КОРИСТУВАЧУ (скопіюй у відповідь і у save_report) >>>")
    lines.append("### Джерела (PDF):")
    lines.extend(citation_lines)
    lines.append("<<< КІНЕЦЬ БЛОКУ ДЖЕРЕЛ >>>")

    return "\n".join(lines)


# ── Entrypoint ──────────────────────────────────────────────────

if __name__ == "__main__":
    port = settings.search_mcp_port
    print(f"🔍 SearchMCP starting on port {port}...")
    mcp.run(transport="streamable-http", host="127.0.0.1", port=port)
