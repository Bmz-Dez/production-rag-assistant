"""
Load documents, split them into chunks, embed them, and save to Chroma.

Run with:
    python -m src.ingest
"""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import settings


def load_documents(data_dir: Path) -> list[Document]:
    """Read PDF, Markdown, and text files from the data folder."""
    documents = []

    for path in data_dir.rglob("*"):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()

        try:
            if suffix == ".pdf":
                docs = PyPDFLoader(str(path)).load()
            elif suffix in {".md", ".txt"}:
                docs = TextLoader(str(path), encoding="utf-8").load()
            else:
                continue
        except Exception as error:
            print(f"Could not load {path.name}: {error}")
            continue

        documents.extend(docs)
        print(f"Loaded: {path.name}")

    print(f"Loaded {len(documents)} raw documents")
    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split long documents into smaller pieces and keep the file name."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)

    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        chunk.metadata["source"] = Path(source).name

    print(f"Created {len(chunks)} chunks")
    return chunks


def build_vectorstore(chunks: list[Document]) -> Chroma:
    """Create embeddings locally (free) and save them on disk."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(settings.persist_dir),
        collection_name="rag_docs",
    )
    print(f"Vector store saved to {settings.persist_dir}")
    return vectorstore


def main():
    if not settings.data_dir.exists():
        settings.data_dir.mkdir(parents=True)
        print(f"Created {settings.data_dir}. Add files there and run again.")
        return

    docs = load_documents(settings.data_dir)
    if not docs:
        print("No documents found in data/. Add some files and try again.")
        return

    chunks = chunk_documents(docs)
    build_vectorstore(chunks)
    print("Ingestion complete.")


if __name__ == "__main__":
    main()