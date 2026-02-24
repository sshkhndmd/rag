## Запуск backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python -m uvicorn app:app --reload --port 8000
```

## Запуск frontend

```powershell
cd frontend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
streamlit run app.py
```

## Что умеет проект

- Загрузка PDF через UI (Streamlit)
- Индексация PDF в ChromaDB
- Разбиение текста на чанки (chunking)
- Векторизация (embeddings)
- Поиск похожих фрагментов (similarity search)
- Вывод страниц-источников и найденных фрагментов

---

## Архитектура проекта

- **Frontend**: `Streamlit`
- **Backend**: `FastAPI`
- **Vector DB**: `ChromaDB`
- **Embeddings**: `HuggingFace` (`sentence-transformers/all-MiniLM-L6-v2`) или `OpenAI` (опционально)
- **RAG-логика**: `LangChain`
