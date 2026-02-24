import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="RAG бот по конспектам", page_icon="📚", layout="wide")
st.title("📚 Чат-бот по конспектам курса (RAG)")
st.caption("Streamlit фронтенд → FastAPI бэкенд (ChromaDB + embeddings + retrieval).")

with st.sidebar:
    st.header("🔌 Backend")
    st.write("URL:", BACKEND_URL)

    if st.button("Проверить /health", use_container_width=True):
        try:
            r = requests.get(f"{BACKEND_URL}/health", timeout=10)
            r.raise_for_status()
            st.json(r.json())
        except Exception as e:
            st.error(f"Не удалось подключиться: {e}")

    st.markdown("---")
    st.header("📄 Загрузка PDF")

    uploaded_pdf = st.file_uploader(
        "Выбери PDF-конспект",
        type=["pdf"],
        accept_multiple_files=False
    )

    if uploaded_pdf is not None:
        st.write(f"Файл: **{uploaded_pdf.name}**")
        if st.button("⬆️ Загрузить и проиндексировать", use_container_width=True):
            try:
                files = {
                    "file": (uploaded_pdf.name, uploaded_pdf.getvalue(), "application/pdf")
                }
                with st.spinner("Загрузка и индексация PDF..."):
                    r = requests.post(
                        f"{BACKEND_URL}/upload_pdf",
                        files=files,
                        timeout=180,
                    )
                    r.raise_for_status()
                    data = r.json()
                st.success(
                    f"Готово! Файл {data.get('filename')} загружен. "
                    f"Добавлено чанков: {data.get('chunks_added')}"
                )
                st.json(data)
            except Exception as e:
                st.error(f"Ошибка загрузки/индексации: {e}")

question = st.text_input(
    "Вопрос по конспекту",
    placeholder="Например: Объясни backpropagation простыми словами"
)

col1, col2 = st.columns([1, 1])
ask_btn = col1.button("🔎 Спросить", use_container_width=True)
clear_btn = col2.button("🧹 Очистить", use_container_width=True)

if clear_btn:
    st.session_state.pop("answer", None)
    st.session_state.pop("sources", None)
    st.rerun()

if ask_btn and question.strip():
    with st.spinner("Запрос в backend..."):
        try:
            r = requests.post(
                f"{BACKEND_URL}/ask",
                json={"question": question.strip()},
                timeout=60,
            )
            r.raise_for_status()
            data = r.json()
            st.session_state["answer"] = data["answer"]
            st.session_state["sources"] = data["sources"]
        except Exception as e:
            st.error(f"Ошибка запроса: {e}")

if "answer" in st.session_state:
    st.subheader("✅ Ответ")
    st.write(st.session_state["answer"])

if "sources" in st.session_state:
    st.subheader("📌 Источники")
    for i, s in enumerate(st.session_state["sources"], start=1):
        src = os.path.basename(s.get("source", "unknown"))
        page = s.get("page", None)
        page_str = f"{page + 1}" if isinstance(page, int) else "?"
        chunk_id = s.get("chunk_id", "?")

        with st.expander(f"[{i}] {src} — стр. {page_str} — chunk {chunk_id}"):
            st.write(s.get("text", ""))