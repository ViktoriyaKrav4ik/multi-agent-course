"""
Перегляд «структури» локального RAG-індексу (FAISS + chunks для BM25).

Запуск з каталогу homework-lesson-5:
    python inspect_index.py
    python inspect_index.py --sample 5
    python inspect_index.py --text-vector 5   # текст ↔ перші числа вектора (якщо FAISS підтримує reconstruct)

Потрібні ті самі .env та ingest, що й для агента.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

from config import Settings, get_rag_index_path
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect FAISS + chunks.pkl structure")
    parser.add_argument(
        "--sample",
        type=int,
        default=3,
        help="Скільки прикладів чанків показати (за замовчуванням 3)",
    )
    parser.add_argument(
        "--text-vector",
        type=int,
        metavar="N",
        default=0,
        help="Показати відповідність «текст чанка → вектор (перші координати)» для N записів (0 = вимкнено)",
    )
    args = parser.parse_args()

    settings = Settings()
    index_dir = get_rag_index_path(settings)
    chunks_path = index_dir / "chunks.pkl"

    print("=== Файли на диску (векторна частина + BM25) ===\n")
    print(f"Каталог індексу: {index_dir}\n")
    if not index_dir.is_dir():
        print("Каталог не існує. Спочатку: python ingest.py")
        sys.exit(1)

    for p in sorted(index_dir.iterdir()):
        if p.is_file():
            print(f"  {p.name:30}  {p.stat().st_size:>12,} bytes")

    print(
        """
Що це:
  index.faiss  — бінарний індекс FAISS (вектори + структура пошуку nearest neighbors).
  index.pkl    — у LangChain тут docstore: id → текст чанка + metadata (сторінка, файл…).
  chunks.pkl   — ті самі Document-и списком для BM25 (лексичний пошук).
"""
    )

    if not chunks_path.is_file():
        print("Немає chunks.pkl — перезапустіть ingest.py")
        sys.exit(1)

    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    print("=== chunks.pkl (для BM25) ===\n")
    print(f"Кількість чанків: {len(chunks)}")
    n = min(args.sample, len(chunks))
    for i in range(n):
        d = chunks[i]
        meta = getattr(d, "metadata", None) or {}
        text = (getattr(d, "page_content", "") or "")[:200].replace("\n", " ")
        print(f"\n--- Чанк {i} ---")
        print(f"metadata: {meta}")
        print(f"текст (початок): {text}...")

    print("\n=== FAISS у пам’яті (після завантаження) ===\n")
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.api_key.get_secret_value(),
    )
    vs = FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    idx = vs.index
    nvec = getattr(idx, "ntotal", None)
    dim = getattr(idx, "d", None)
    print(f"Векторів у індексі (ntotal): {nvec}")
    print(f"Розмірність вектора (d):     {dim}")
    n_docs = len(getattr(vs, "index_to_docstore_id", {}) or {})
    print(f"Зв'язок вектор→id→текст:     {n_docs} записів")
    print(
        "\nЛогічна структура: кожен чанк → один embedding-вектор довжини `d`; "
        "FAISS шукає найближчі вектори до embedding запиту; за id дістаємо текст із docstore."
    )

    # Текст ↔ вектор: вектор фізично в index.faiss; текст у docstore; зв’язок — позиція i в індексі.
    if args.text_vector and nvec:
        k = min(args.text_vector, int(nvec))
        mapping = getattr(vs, "index_to_docstore_id", None) or {}
        print(f"\n=== Відповідність текст ↔ вектор (перші {k} векторів у FAISS) ===\n")
        has_reconstruct = callable(getattr(idx, "reconstruct", None))
        if not has_reconstruct:
            print(
                "Цей тип FAISS-індексу не підтримує reconstruct() — "
                "вектор як масив з диску не показати; логіка все одно: один рядок індексу = один чанк."
            )
        for i in range(k):
            doc_id = None
            if isinstance(mapping, dict):
                doc_id = mapping.get(i)
            elif isinstance(mapping, (list, tuple)) and i < len(mapping):
                doc_id = mapping[i]
            text = ""
            meta = {}
            if doc_id is not None:
                try:
                    doc = vs.docstore.search(doc_id)
                    text = (getattr(doc, "page_content", "") or "")[:180].replace("\n", " ")
                    meta = getattr(doc, "metadata", None) or {}
                except Exception as e:
                    text = f"(не вдалося прочитати docstore: {e})"
            print(f"\n--- Позиція в індексі i = {i} ---")
            print(f"docstore id: {doc_id!r}")
            print(f"metadata: {meta}")
            print(f"текст (початок): {text}...")
            if has_reconstruct:
                try:
                    vec = idx.reconstruct(int(i))
                    head = [round(float(x), 6) for x in vec[:12]]
                    print(f"вектор: dim={len(vec)}, перші 12 координат: {head} …")
                except Exception as e:
                    print(f"reconstruct({i}): {e}")


if __name__ == "__main__":
    main()
