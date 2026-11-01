"""
Search the saved document chunks.

Hybrid retrieval = vector search + keyword search (BM25).
"""

from functools import lru_cache
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document

from src.config import settings


@lru_cache(maxsize=1)
def get_embeddings():
    """Same free embedding model we used during ingestion (cached in memory)."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def get_vectorstore() -> Chroma:
    """Open the database created by ingest.py."""
    return Chroma(
        persist_directory=str(settings.persist_dir),
        embedding_function=get_embeddings(),
        collection_name="rag_docs",
    )


def get_hybrid_retriever(k: int | None = None):
    """Combine vector search and keyword search."""
    k = k or settings.top_k
    vectorstore = get_vectorstore()

    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    stored = vectorstore.get()
    docs_for_bm25 = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(stored["documents"], stored["metadatas"])
    ]

    bm25_retriever = BM25Retriever.from_documents(docs_for_bm25)
    bm25_retriever.k = k

    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.6, 0.4],
    )


def format_docs_with_citations(docs: list[Document]) -> str:
    """Turn retrieved chunks into text the chat model can cite."""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        page_str = f", page {page}" if page != "" else ""
        parts.append(f"[{i}] Source: {source}{page_str}\n{doc.page_content}")
    return "\n\n".join(parts)