"""Інструменти Research Agent: пошук, читання URL, збереження звіту."""

import os
from contextlib import redirect_stderr
from datetime import datetime
from io import StringIO

from langchain_core.tools import tool

from config import Settings

settings = Settings()


@tool
def web_search(query: str) -> str:
    """Пошук в інтернеті за запитом. Повертає список результатів: заголовок, посилання, короткий сніпет."""
    try:
        from ddgs import DDGS
    except ImportError:
        return "Помилка: бібліотека ddgs не встановлена. Запусти: pip install ddgs"

    try:
        # Придушуємо попередження ddgs про "Impersonate ... does not exist, using 'random'"
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


@tool
def read_url(url: str) -> str:
    """Отримати основний текст сторінки за URL. Підходить для статей та довгих сторінок."""
    try:
        import trafilatura
    except ImportError:
        return "Помилка: бібліотека trafilatura не встановлена. Запусти: pip install trafilatura"

    if not url.startswith(("http://", "https://")):
        return "Помилка: невалідний URL (має починатися з http:// або https://)."

    try:
        downloaded = trafilatura.fetch_url(url)
    except Exception as e:
        return f"Помилка завантаження URL: {e}"

    if not downloaded:
        return "Не вдалося завантажити сторінку (таймаут, 404 або сайт недоступний)."

    text = trafilatura.extract(downloaded)
    if not text or not text.strip():
        return "Не вдалося витягнути текст із сторінки."

    max_len = settings.max_url_content_length
    if len(text) > max_len:
        text = text[:max_len] + "\n\n[... текст обрізано ...]"
    return text


@tool
def write_report(filename: str, content: str) -> str:
    """Зберегти Markdown-звіт у файл. filename — лише ім'я файлу (наприклад report.md), content — текст звіту в Markdown."""
    if not filename.strip():
        return "Помилка: вкажи непусте ім'я файлу."
    filename = filename.strip()
    if not filename.endswith(".md"):
        filename = filename + ".md"
    # Якщо агент передав report.md — робимо унікальне ім'я (час), щоб не перезаписувати
    if filename.lower() == "report.md":
        filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

    # Абсолютний шлях відносно папки homework-lesson-3 — файл завжди в правильному місці
    _root = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(_root, settings.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)

    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Звіт збережено: {path}"
    except Exception as e:
        return f"Помилка збереження файлу: {e}"
