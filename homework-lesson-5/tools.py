"""Інструменти Research Agent: веб, URL, звіт, локальний RAG."""

from __future__ import annotations

import os
from contextlib import redirect_stderr
from datetime import datetime
from io import StringIO

from langchain_core.tools import tool

from config import Settings

settings = Settings()


@tool
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
    return "\n\n".join(lines)


@tool
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


@tool
def write_report(filename: str, content: str) -> str:
    """Зберегти Markdown-звіт у output/. filename — ім'я файлу (.md)."""
    if not filename.strip():
        return "Помилка: вкажіть ім'я файлу."
    filename = filename.strip()
    if not filename.endswith(".md"):
        filename = filename + ".md"
    if filename.lower() == "report.md":
        filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

    root = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(root, settings.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)

    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Звіт збережено: {path}"
    except Exception as e:
        return f"Помилка збереження файлу: {e}"


@tool
def knowledge_search(query: str) -> str:
    """Пошук у локальній базі знань (індексовані PDF/текст з data/). Hybrid + reranking."""
    from retriever import hybrid_search, index_ready

    if not index_ready():
        return (
            "Локальний індекс ще не створено. З каталогу homework-lesson-5 виконайте: "
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
                page_human = int(page) + 1  # PyPDF: 0-based → показуємо 1-based
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
    lines.append("<<< ДЛЯ ВІДПОВІДІ КОРИСТУВАЧУ (скопіюй у відповідь і у write_report) >>>")
    lines.append("### Джерела (PDF):")
    lines.extend(citation_lines)
    lines.append("<<< КІНЕЦЬ БЛОКУ ДЖЕРЕЛ >>>")

    return "\n".join(lines)
