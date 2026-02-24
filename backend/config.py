import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Settings:
    chroma_dir: str
    collection_name: str
    chunk_size: int
    chunk_overlap: int
    top_k: int

    embeddings_provider: str
    openai_model: str
    openai_embedding_model: str
    hf_embedding_model: str

def get_settings() -> Settings:
    return Settings(
        chroma_dir=os.getenv("CHROMA_DIR", "chroma_db"),
        collection_name=os.getenv("COLLECTION_NAME", "course_notes"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "900")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
        top_k=int(os.getenv("TOP_K", "3")),
        embeddings_provider=os.getenv("EMBEDDINGS_PROVIDER", "openai").lower().strip(),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        hf_embedding_model=os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
    )
