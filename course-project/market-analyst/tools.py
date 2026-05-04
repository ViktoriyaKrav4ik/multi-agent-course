"""Веб-пошук, RAG, read_url для Analyst / Critic."""

from __future__ import annotations

import json
import os
import re
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

from langchain_core.tools import tool

from config import Settings

_settings: Settings | None = None


def _settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


@tool
def web_search(query: str) -> str:
    """DuckDuckGo: заголовки, URL, сніпети."""
    settings = _settings()
    try:
        from ddgs import DDGS
    except ImportError:
        return "Помилка: pip install ddgs"

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
        text = text[:max_len] + "\n\n[... обрізано ...]"
    return text


@tool
def read_url(url: str) -> str:
    """Текст сторінки за URL (верифікація)."""
    settings = _settings()
    try:
        import trafilatura
    except ImportError:
        return "Помилка: pip install trafilatura"

    if not url.startswith(("http://", "https://")):
        return "URL має починатися з http(s)://"

    try:
        downloaded = trafilatura.fetch_url(url)
    except Exception as e:
        return f"Помилка завантаження: {e}"

    if not downloaded:
        return "Не вдалося завантажити сторінку."

    text = trafilatura.extract(downloaded)
    if not text or not text.strip():
        return "Не вдалося витягнути текст."

    max_len = settings.max_url_content_length
    if len(text) > max_len:
        text = text[:max_len] + "\n\n[... обрізано ...]"
    return text


@tool
def knowledge_search(query: str) -> str:
    """Локальний корпус (corpus/ після ingest). Тільки для Analyst згідно ТЗ."""
    from retriever import hybrid_search, index_ready

    if not index_ready():
        return (
            "Індекс не зібрано. Виконайте: python ingest.py (файли в corpus/)."
        )

    settings = _settings()
    try:
        docs = hybrid_search(query)
    except Exception as e:
        return f"Помилка knowledge_search: {e}"

    if not docs:
        return "У локальному корпусі релевантного немає."

    lines: list[str] = [
        "Фрагменти з корпусу (цитуй file + сторінку якщо є).\n",
        f"Кількість: {len(docs)}\n",
    ]
    cites: list[str] = []

    for i, doc in enumerate(docs, 1):
        meta = doc.metadata or {}
        src = meta.get("source") or meta.get("file_name", "невідомо")
        if isinstance(src, str) and len(src) > 80:
            src = os.path.basename(src)

        page = meta.get("page")
        if page is not None:
            try:
                page_h = int(page) + 1
                page_line = f"Сторінка PDF: {page_h}"
                cite_p = str(page_h)
            except (TypeError, ValueError):
                page_line = f"Сторінка: {page}"
                cite_p = str(page)
        else:
            page_line = "Сторінка: н/д"
            cite_p = "н/д"

        cites.append(f"- {src}, стор. {cite_p}")
        snippet = (doc.page_content or "").strip()
        if len(snippet) > 1200:
            snippet = snippet[:1200] + "\n[...]"
        lines.append(
            f"---\n### Фрагмент {i}\n- Файл: {src}\n- {page_line}\n\n{snippet}"
        )

    lines.extend(["", "Джерела (корпус):", *cites])
    return "\n".join(lines)


def _pick_col(df_cols: list[str], aliases: list[str]) -> str | None:
    lower_map = {c.strip().lower(): c for c in df_cols}
    for a in aliases:
        if a.lower() in lower_map:
            return lower_map[a.lower()]
    return None


def _read_csv_flexible(path: Path):
    import pandas as pd

    # Найчастіші формати експорту з Excel/PowerBI: ; або ,, інколи з технічним першим рядком.
    attempts = [
        {"sep": ";", "skiprows": 0},
        {"sep": ";", "skiprows": 1},
        {"sep": ",", "skiprows": 0},
        {"sep": ",", "skiprows": 1},
    ]
    for opt in attempts:
        try:
            df = pd.read_csv(path, **opt)
            if df.shape[1] > 1:
                cols = {str(c).strip().lower() for c in df.columns}
                if "hybrid" in cols or "гібрид" in cols:
                    return df
        except Exception:
            continue
    # fallback
    return pd.read_csv(path)


def _find_csv_path(csv_path: str | None) -> Path | None:
    root = Path(__file__).resolve().parent
    if csv_path:
        p = Path(csv_path)
        if not p.is_absolute():
            p = (root / p).resolve()
        return p if p.is_file() else None

    corpus = root / "corpus"
    candidates = sorted(corpus.rglob("*.csv"), key=lambda x: x.stat().st_size, reverse=True)
    return candidates[0] if candidates else None


@tool
def rank_corn_hybrids(
    objective: str = "balanced",
    top_n: int = 10,
    max_moisture: float = 22.0,
    csv_path: str = "",
) -> str:
    """
    Ранжування гібридів кукурудзи з CSV (детерміновано, без LLM).
    objective: balanced | ebitda | yield
    """
    try:
        import pandas as pd
    except ImportError:
        return "Помилка: встановіть pandas (pip install pandas)."

    p = _find_csv_path(csv_path or None)
    if p is None:
        return "CSV не знайдено. Покладіть файл у corpus/ (або передайте csv_path)."

    try:
        df = _read_csv_flexible(p)
    except Exception as e:
        return f"Не вдалося прочитати CSV: {e}"

    if df.empty:
        return "CSV порожній."

    cols = list(df.columns)
    hybrid_col = _pick_col(cols, ["Hybrid", "Hybid", "Гібрид", "гібрид"])
    year_col = _pick_col(cols, ["Year", "Рік", "year"])
    loc_col = _pick_col(cols, ["Village", "Локація", "Cluster_Village", "location"])
    yield_col = _pick_col(cols, ["Урожайність", "Yield", "yield_t_ha"])
    moisture_col = _pick_col(cols, ["Вологість", "Moisture", "moisture_harvest"])
    ebitda_col = _pick_col(cols, ["EBITDA", "ebitda"])

    required = [hybrid_col, yield_col, moisture_col, ebitda_col]
    if any(c is None for c in required):
        return (
            "У CSV не знайдено потрібні колонки. Очікуються: Hybrid/Гібрид, "
            "Урожайність, Вологість, EBITDA."
        )

    work = df.copy()
    for c in [yield_col, moisture_col, ebitda_col]:
        # Підтримка десяткової коми: 14,2 -> 14.2
        work[c] = (
            work[c]
            .astype(str)
            .str.replace(" ", "", regex=False)
            .str.replace(",", ".", regex=False)
            .str.replace("%", "", regex=False)
        )
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna(subset=[hybrid_col, yield_col, moisture_col, ebitda_col])
    if work.empty:
        return "Після очистки немає валідних рядків."

    grouped = (
        work.groupby(hybrid_col)
        .agg(
            n_obs=(hybrid_col, "size"),
            yield_mean=(yield_col, "mean"),
            yield_std=(yield_col, "std"),
            moisture_mean=(moisture_col, "mean"),
            ebitda_mean=(ebitda_col, "mean"),
        )
        .reset_index()
    )
    grouped["yield_std"] = grouped["yield_std"].fillna(0.0)
    grouped["moisture_penalty"] = (grouped["moisture_mean"] - float(max_moisture)).clip(lower=0)
    grouped["stability_penalty"] = grouped["yield_std"]

    # Min-max scaling для стабільного скорингу
    def minmax(col: str) -> list[float]:
        mn, mx = grouped[col].min(), grouped[col].max()
        if mx == mn:
            return [0.5] * len(grouped)
        return ((grouped[col] - mn) / (mx - mn)).tolist()

    grouped["yield_score"] = minmax("yield_mean")
    grouped["ebitda_score"] = minmax("ebitda_mean")
    grouped["moisture_penalty_score"] = minmax("moisture_penalty")
    grouped["stability_penalty_score"] = minmax("stability_penalty")

    obj = (objective or "balanced").strip().lower()
    if obj == "yield":
        w = dict(y=0.7, e=0.15, m=0.1, s=0.05)
    elif obj == "ebitda":
        w = dict(y=0.15, e=0.7, m=0.1, s=0.05)
    else:
        w = dict(y=0.4, e=0.4, m=0.15, s=0.05)

    grouped["total_score"] = (
        w["y"] * grouped["yield_score"]
        + w["e"] * grouped["ebitda_score"]
        - w["m"] * grouped["moisture_penalty_score"]
        - w["s"] * grouped["stability_penalty_score"]
    )
    ranked = grouped.sort_values("total_score", ascending=False).head(max(1, int(top_n)))

    meta = {
        "csv_source": str(p),
        "objective": obj,
        "max_moisture": max_moisture,
        "rows_used": int(len(work)),
        "unique_hybrids": int(work[hybrid_col].nunique()),
        "year_coverage": sorted(work[year_col].dropna().unique().tolist()) if year_col else [],
        "location_coverage": int(work[loc_col].nunique()) if loc_col else None,
    }

    items = []
    for _, r in ranked.iterrows():
        items.append(
            {
                "hybrid": r[hybrid_col],
                "n_obs": int(r["n_obs"]),
                "yield_mean": round(float(r["yield_mean"]), 3),
                "moisture_mean": round(float(r["moisture_mean"]), 3),
                "ebitda_mean": round(float(r["ebitda_mean"]), 3),
                "score": round(float(r["total_score"]), 4),
            }
        )

    return json.dumps({"meta": meta, "ranking": items}, ensure_ascii=False, indent=2)


@tool
def compare_hybrid_years(hybrid_name: str, csv_path: str = "") -> str:
    """Порівняння одного гібриду між роками (yield/moisture/EBITDA)."""
    try:
        import pandas as pd
    except ImportError:
        return "Помилка: встановіть pandas (pip install pandas)."

    name = (hybrid_name or "").strip()
    if not name:
        return "Передайте hybrid_name, наприклад: Гран 6"

    p = _find_csv_path(csv_path or None)
    if p is None:
        return "CSV не знайдено. Покладіть файл у corpus/ (або передайте csv_path)."

    try:
        df = _read_csv_flexible(p)
    except Exception as e:
        return f"Не вдалося прочитати CSV: {e}"

    cols = list(df.columns)
    hybrid_col = _pick_col(cols, ["Hybrid", "Hybid", "Гібрид", "гібрид"])
    year_col = _pick_col(cols, ["Year", "Рік", "year"])
    loc_col = _pick_col(cols, ["Village", "Cluster_Village", "Локація", "location"])
    yield_col = _pick_col(cols, ["Урожайність", "Yield", "yield_t_ha"])
    moisture_col = _pick_col(cols, ["Вологість", "Moisture", "moisture_harvest"])
    ebitda_col = _pick_col(cols, ["EBITDA", "ebitda"])
    if any(c is None for c in [hybrid_col, year_col, yield_col, moisture_col, ebitda_col]):
        return "У CSV не знайдено потрібні колонки для порівняння."

    work = df.copy()
    for c in [yield_col, moisture_col, ebitda_col]:
        work[c] = (
            work[c]
            .astype(str)
            .str.replace(" ", "", regex=False)
            .str.replace(",", ".", regex=False)
            .str.replace("%", "", regex=False)
        )
        work[c] = pd.to_numeric(work[c], errors="coerce")

    work = work.dropna(subset=[hybrid_col, year_col, yield_col, moisture_col, ebitda_col])
    if work.empty:
        return "Немає валідних рядків після очистки."

    mask = work[hybrid_col].astype(str).str.strip().str.lower() == name.lower()
    sel = work.loc[mask].copy()
    if sel.empty:
        # fallback: contains
        mask2 = work[hybrid_col].astype(str).str.lower().str.contains(re.escape(name.lower()))
        sel = work.loc[mask2].copy()
    if sel.empty:
        return f"Гібрид '{name}' не знайдено в CSV."

    by_year = (
        sel.groupby(year_col)
        .agg(
            n_obs=(hybrid_col, "size"),
            yield_mean=(yield_col, "mean"),
            moisture_mean=(moisture_col, "mean"),
            ebitda_mean=(ebitda_col, "mean"),
        )
        .reset_index()
        .sort_values(year_col)
    )

    by_loc_year = None
    if loc_col:
        by_loc_year = (
            sel.groupby([year_col, loc_col])
            .agg(
                n_obs=(hybrid_col, "size"),
                yield_mean=(yield_col, "mean"),
                moisture_mean=(moisture_col, "mean"),
                ebitda_mean=(ebitda_col, "mean"),
            )
            .reset_index()
            .sort_values([year_col, loc_col])
        )

    out = {
        "meta": {
            "csv_source": str(p),
            "hybrid": name,
            "rows_used": int(len(sel)),
            "years": [int(y) if str(y).isdigit() else str(y) for y in by_year[year_col].tolist()],
        },
        "by_year": by_year.to_dict(orient="records"),
        "by_location_year": by_loc_year.to_dict(orient="records") if by_loc_year is not None else [],
    }
    return json.dumps(out, ensure_ascii=False, indent=2)
