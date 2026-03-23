"""
Knowledge ingestion pipeline.

Usage (з кореня homework-lesson-5): python ingest.py
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Settings, get_rag_index_path

CHUNKS_FILENAME = "chunks.pkl"


def _load_documents(data_dir: Path) -> list:
    from langchain_core.documents import Document

    documents: list[Document] = []
    if not data_dir.is_dir():
        return documents

    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        try:
            if suffix == ".pdf":
                documents.extend(PyPDFLoader(str(path)).load())
            elif suffix in (".txt", ".md"):
                documents.extend(
                    TextLoader(str(path), encoding="utf-8").load()
                )
        except Exception as e:
            print(f"⚠️  Пропуск {path}: {e}")
    return documents


def ingest() -> None:
    settings = Settings()
    root = Path(__file__).resolve().parent
    data_path = root / settings.data_dir
    index_path = get_rag_index_path(settings)

    if not (settings.rag_index_dir or os.environ.get("RAG_INDEX_DIR", "") or "").strip():
        try:
            str((root / settings.index_dir).resolve()).encode("ascii")
        except UnicodeEncodeError:
            print(
                "⚠️  Шлях до проєкту містить не-ASCII символи; FAISS на Windows у такому "
                "шляху часто не записує файли.\n"
                f"   Індекс зберігається тут: {index_path}\n"
                "   (або задайте RAG_INDEX_DIR у .env / змінних середовища)\n"
            )

    print(f"📂 Джерело: {data_path}")
    documents = _load_documents(data_path)
    if not documents:
        print(
            "❌ Документів не знайдено. Додайте PDF, TXT або MD у папку data/ "
            "і запустіть знову."
        )
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    print(f"✂️  Чанків: {len(chunks)}")

    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.api_key.get_secret_value(),
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)

    index_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_path))

    chunks_file = index_path / CHUNKS_FILENAME
    with open(chunks_file, "wb") as f:
        pickle.dump(chunks, f)

    print(f"✅ FAISS індекс збережено: {index_path}")
    print(f"✅ Чанки для BM25: {chunks_file}")


if __name__ == "__main__":
    ingest()
