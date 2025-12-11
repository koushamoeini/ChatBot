# Chatbot Project

Quick start (Windows / PowerShell):

1. Create & activate a virtual environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
2. Install dependencies (root-level `requirements.txt`):
```powershell
pip install -r requirements.txt
```
3. Run the backend Django server:
```powershell
# (Optional) Set the API key before running the server
$env:API_KEY = '<YOUR_OPENAI_COMPAT_API_KEY>'
python manage.py migrate
python manage.py runserver 0.0.0.0:8000
```
4. Open the frontend in your browser (Django serves it):
```powershell
# Django will serve the frontend at http://127.0.0.1:8000
```

Or run the ingestion notebook:
1. Start Jupyter:
```powershell
python -m jupyter notebook
```
2. Open `ingest.ipynb`, ensure the venv is active, and run the cells to populate `chroma_db`.

The API key is read from the `API_KEY` environment variable (fallback default in `chatbot_app/views.py`). Set `API_KEY` before running the server for production deployments.

Configuration and Environment Variables
-------------------------------------
- `API_KEY` (required): Your OpenAI-compatible API key used by the LLM client.
- `RELEVANCE_THRESHOLD` (optional): Float between 0 and 1 for the cosine similarity filter (default 0.6). Higher means stricter matching. To set in PowerShell:
```powershell
$env:RELEVANCE_THRESHOLD = '0.6'
```

Populating the Vector DB (ChromaDB)
---------------------------------
The chat relies on a ChromaDB vector store for retrieval. If the collection is empty, you will get "اطلاعات کافی نیست" responses.

1. Ensure your `knowledge_base.json` is prepared (each entry has `chunk_id`, `chunk_text`, `metadata`).
2. Start Jupyter Notebook with the same virtual environment and open `ingest.ipynb`.
```powershell
python -m jupyter notebook
```
3. Run the cells in `ingest.ipynb` to create/overwrite the `farsi_rag_collection` in `./chroma_db`.

Agentic RAG behavior (ReAct-style)
---------------------------------
This project uses an "agentic" RAG flow: the model can call small tools (retrieve and summarize) in a few steps and then produce a final answer. To reduce hallucinations:

- Retrieval uses sentence-transformer embeddings and a cosine similarity filter (controlled by `RELEVANCE_THRESHOLD`) that rejects irrelevant chunks.
- Final answers are validated against the retrieved context: if the LLM's answer isn't supported by the retrieved chunks, the system returns a polite "no data" message instead of hallucinating an answer.
- The bot now replies to simple greetings directly (e.g., "سلام", "خوبی؟").
- The validation is slightly relaxed to avoid rejecting low-confidence but useful answers; if only weakly-related documents are available, the bot will give a `low-confidence` best-effort reply prefixed with a small disclaimer rather than always returning "no data".

Usage Example (HTTP API)
------------------------
Endpoint: `POST /ask`
Request Body:
```json
{
    "question": "اطلاعات درباره سرویس X چیست؟",
    "history": [{"role":"user","content":"سوال قبلی"}, {"role":"assistant","content":"پاسخ"}]
}
```
Response (success):
```json
{
    "answer": "پاسخ معتبر و پشتیبانی‌شده توسط متن‌های موجود..."
}
```
If there is not enough relevant information: the server responds with a user-friendly message like:
```json
{ "answer": "متاسفانه اطلاعات کافی برای پاسخ به این سوال در پایگاه دانش موجود نیست." }
```
Optional `allow_general_knowledge` parameter:
```json
{ "question": "Who invented the lightbulb?", "allow_general_knowledge": true }
```
If `allow_general_knowledge` is set to `true`, the agent will attempt to answer using its general knowledge (LLM) when the KB doesn't contain supporting evidence, and will clearly prefix the response with a short disclaimer stating it's best-effort.

Troubleshooting
---------------
- If you see `ModuleNotFoundError: No module named 'chatbot'`, ensure you're running from repo root and that `backend` is a package (it is by default in this repo). Also use Django runserver (`python manage.py runserver`) instead of uvicorn.
- If `chroma_db` is empty or collection `farsi_rag_collection` doesn't exist, run the ingestion (`ingest.ipynb`) to populate the DB.
- If answers look too general/hallucinated, increase `RELEVANCE_THRESHOLD` or verify `chroma_db` contains valid chunks for the question.

Security and production
-----------------------
- Do not store `API_KEY` directly in the repository. Use environment variables or secret management services.
- For production deployments, configure a proper WSGI/ASGI server (e.g., `gunicorn` + `nginx`) and secure the API with authentication and HTTPS.

Contributing and Tests
----------------------
- This repository contains `ingest.ipynb` to populate vectors. For contributions, please ensure your changes include tests or validation when possible and avoid committing sensitive data like API keys.


This repository contains the code for a chatbot, developed across multiple phases, evolving from a simple keyword-based system to a more sophisticated RAG application with a web interface.

## Description

A Python-based RAG (Retrieval-Augmented Generation) chatbot designed to answer questions based on a provided knowledge base. It leverages vector embeddings for semantic search, integrates with OpenAI-compatible language models, and features a Django backend with a web frontend. 🗣️ 🌐

---

## Accessing Project Phases

This project uses Git tags to mark the end of each development phase.

* **Phase 1 (`phase-1-cli`)**: The initial baseline version. 🌱
    * **Knowledge Source:** Loads text directly from `.docx` files located in a `knowledge_docs` directory.
    * **Chunking:** Splits documents into chunks based on blank lines (double newlines) within the `.docx` files.
    * **Retrieval Method:** Implements a **naive keyword matching** strategy. It splits the query and chunks into words and finds the chunk with the highest number of overlapping words. Only the single best-matching chunk is used as context.
    * **Response Generation:** Uses the `openai` library, configured to potentially connect to custom endpoints (like `api.metisai.ir`), to generate answers based on the retrieved chunk and the user's query. The prompt includes basic instructions for the LLM to act as a customer service assistant.
    * **User Interface:** Interaction occurs directly within the `RAG_Chatbot.ipynb` Jupyter Notebook through an input loop.
    * **Dependencies:** `openai`, `python-docx`, `python-dotenv`.

* **Phase 2 (`phase-2`)**: Enhanced retrieval, structured data, and CLI. 🚀
    * **Knowledge Source:** Shifts to a structured `knowledge_base.json` file, where each entry contains a `chunk_id`, `chunk_text`, and associated `metadata` (like source title and URL).
    * **Ingestion Pipeline:** Introduces a dedicated ingestion script (`Vector_Database_Ingestion.ipynb`). It uses `sentence-transformers` (`paraphrase-multilingual-mpnet-base-v2`) to generate vector embeddings for each chunk's text.
    * **Vector Database:** Stores the embeddings, original text, and metadata persistently in a **`ChromaDB`** vector store located in the `./chroma_db` directory.
    * **Retrieval Method:** Implements **semantic search**. Queries are embedded using the same `sentence-transformers` model, and `ChromaDB` retrieves the `n_results` (e.g., 10) most similar chunks based on vector distance.
    * **Query Enhancement:** Adds an **LLM-based query correction/expansion** step (`expand_query`). It uses `gpt-4o-mini` (via `api.tapsage.com`) to correct spelling/grammar mistakes in the user's query. Retrieval is performed using *both* the original and corrected queries to broaden the search.
    * **User Interface:** Provides a dedicated **Command-Line Interface (CLI)** application (`Chatbot_CLI_Application.ipynb`) for a more standard user interaction flow.
    * **Dependencies:** `chromadb`, `sentence-transformers`, `openai`, `python-dotenv`.

* **Phase 3 (`phase-3`)**: Backend API, Web UI, and Conversation History. ✨
    * **Architecture:** Implements a client-server model.
    * **Backend:** A **Django** application (Django project in `web/` and `chatbot_app`) serves the chatbot logic via an `/ask` REST endpoint. It encapsulates the core chatbot functions within a `RAGChatbot` class (`backend/chatbot.py`). Django serves the frontend template and serves the static assets during development. The RAG pipeline now supports an "agentic" ReAct-style flow where the model can call small tools (retrieve/summarize) and then produce a final answer.
    * **Frontend:** A simple **Web UI** (`templates/index.html`) using HTML and vanilla JavaScript allows users to interact with the chatbot through their browser. It communicates with the Django backend via `fetch` requests. Includes basic chat styling, message display, and a typing indicator.
    * **Conversation Context:** Introduces **query rewriting** (`rewrite_query` in `backend/chatbot.py`). Before retrieval, the LLM reframes the user's latest question using the recent conversation history (last few turns) to make it a standalone, contextually complete query.
    * **Ingestion Improvement:** The ingestion script (`ingest.ipynb`) now **enriches the text *before* embedding**. It prepends metadata (like source title and parent category) to the chunk text to potentially create more contextually rich embeddings for better retrieval, while still storing the original text in the database for display/generation.
    * **Generation Enhancement:** The prompt sent to the LLM for the final answer now includes formatted context chunks *with their metadata* (source, URL) and the recent conversation history, allowing for more informed and potentially source-citing responses.
    * **Dependencies:** Managed via `requirements.txt`, including `django`, `gunicorn`, `chromadb`, `sentence-transformers`, `openai`, `python-dotenv`.
