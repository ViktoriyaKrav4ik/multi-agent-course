"""
Hybrid retrieval: semantic (FAISS) + BM25 + cross-encoder reranking.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from config import Settings, get_rag_index_path

CHUNKS_FILENAME = "chunks.pkl"

_retriever_cache: Any = None


def _index_paths(settings: Settings) -> tuple[Path, Path]:
    index_dir = get_rag_index_path(settings)
    chunks_file = index_dir / CHUNKS_FILENAME
    return index_dir, chunks_file


def index_ready() -> bool:
    settings = Settings()
    index_dir, chunks_file = _index_paths(settings)
    faiss_ok = (index_dir / "index.faiss").exists() and (index_dir / "index.pkl").exists()
    return faiss_ok and chunks_file.is_file()


def _build_retriever():
    settings = Settings()
    index_dir, chunks_file = _index_paths(settings)

    if not index_ready():
        raise FileNotFoundError(
            "Індекс не знайдено. Запустіть з папки homework-lesson-8: python ingest.py"
        )

    with open(chunks_file, "rb") as f:
        chunks = pickle.load(f)

    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.api_key.get_secret_value(),
    )
    vectorstore = FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    k_fetch = max(settings.retrieval_top_k, 8)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k_fetch})
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k_fetch

    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=list(settings.ensemble_weights),
    )

    if not settings.use_reranking:
        return ensemble

    cross_encoder = HuggingFaceCrossEncoder(model_name=settings.rerank_model)
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=settings.rerank_top_n)

    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble,
    )


def get_retriever():
    global _retriever_cache
    if _retriever_cache is None:
        _retriever_cache = _build_retriever()
    return _retriever_cache


def hybrid_search(query: str) -> list:
    settings = Settings()
    retriever = get_retriever()
    docs = retriever.invoke(query)
    if not settings.use_reranking and len(docs) > settings.rerank_top_n:
        return docs[: settings.rerank_top_n]
    return docs
