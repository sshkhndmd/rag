import os
import glob
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import get_settings
from rag import get_vectorstore


def load_pdfs(pdf_dir: str) -> List[Document]:
    pdf_paths = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in: {pdf_dir}")

    all_docs: List[Document] = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs = loader.load()  
        all_docs.extend(docs)
    return all_docs


def load_single_pdf(pdf_path: str) -> List[Document]:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    return loader.load()


def split_into_chunks(docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for i, d in enumerate(chunks):
        d.metadata = d.metadata or {}
        d.metadata["chunk_id"] = i
    return chunks


def rebuild_index_all() -> int:
    """
    Полная переиндексация всех PDF из data/pdfs (очистка коллекции).
    Возвращает количество чанков.
    """
    s = get_settings()
    pdf_dir = os.path.join(os.path.dirname(__file__), "data", "pdfs")

    docs = load_pdfs(pdf_dir)
    chunks = split_into_chunks(docs, s.chunk_size, s.chunk_overlap)

    vs = get_vectorstore()

    # Для учебного проекта: чистим коллекцию перед записью
    try:
        vs._collection.delete(where={})
    except Exception:
        pass

    vs.add_documents(chunks)
    try:
        vs.persist()
    except Exception:
        pass

    return len(chunks)


def ingest_one_pdf(pdf_path: str) -> int:
    """
    Индексирует только один PDF (БЕЗ очистки коллекции).
    Возвращает количество добавленных чанков.
    """
    s = get_settings()

    docs = load_single_pdf(pdf_path)
    chunks = split_into_chunks(docs, s.chunk_size, s.chunk_overlap)

    vs = get_vectorstore()
    vs.add_documents(chunks)
    try:
        vs.persist()
    except Exception:
        pass

    return len(chunks)


def main():
    count = rebuild_index_all()
    s = get_settings()
    print(f"✅ Ingest done: {count} chunks -> {s.chroma_dir}/{s.collection_name}")


if __name__ == "__main__":
    main()