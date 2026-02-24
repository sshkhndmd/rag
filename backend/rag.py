import os
import re
from typing import List, Dict, Any, Tuple

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings 

from config import get_settings


def build_embeddings():
    s = get_settings()
    if s.embeddings_provider == "openai":
        return OpenAIEmbeddings(model=s.openai_embedding_model)
    return HuggingFaceEmbeddings(model_name=s.hf_embedding_model)


def get_vectorstore() -> Chroma:
    s = get_settings()
    emb = build_embeddings()
    return Chroma(
        collection_name=s.collection_name,
        persist_directory=s.chroma_dir,
        embedding_function=emb,
    )


def format_sources(docs: List[Document]) -> List[Dict[str, Any]]:
    out = []
    for d in docs:
        md = d.metadata or {}
        out.append(
            {
                "source": md.get("source", "unknown"),
                "page": md.get("page", None),
                "chunk_id": md.get("chunk_id", None),
                "text": d.page_content,
            }
        )
    return out


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _first_sentences(text: str, max_sentences: int = 2, max_chars: int = 320) -> str:
    t = _clean(text)
    parts = re.split(r"(?<=[.!?])\s+", t)
    picked = " ".join(parts[:max_sentences]).strip()
    if len(picked) > max_chars:
        picked = picked[:max_chars].rsplit(" ", 1)[0] + "…"
    return picked


def answer_question(question: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Полностью оффлайн-ответ без OpenAI:
    - retrieval top_k
    - экстрактивная выжимка из найденных фрагментов
    """
    s = get_settings()
    vs = get_vectorstore()

    docs = vs.similarity_search(question, k=s.top_k)
    sources = format_sources(docs)

    if not docs:
        return "Не нашёл релевантных фрагментов в конспекте. Попробуй переформулировать вопрос.", []

    bullets = []
    pages = []
    for i, d in enumerate(docs, start=1):
        src = os.path.basename(d.metadata.get("source", "unknown"))
        page = d.metadata.get("page", None)
        if isinstance(page, int):
            pages.append(page + 1)
            page_str = f"{page + 1}"
        else:
            page_str = "?"
        snippet = _first_sentences(d.page_content, max_sentences=2)
        bullets.append(f"{i}) {snippet} (источник: {src}, стр. {page_str})")

    pages_str = ", ".join(map(str, sorted(set(pages)))) if pages else "?"
    answer = (
        f"По конспекту нашёл такие релевантные фрагменты:\n\n"
        + "\n".join(bullets)
        + f"\n\nИсточники (страницы): {pages_str}"
    )

    return answer, sources
