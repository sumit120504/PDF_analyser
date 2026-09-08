# PDF Research Assistant

A Streamlit application for asking questions about uploaded PDF files. The app extracts text from a PDF, splits it into searchable chunks, retrieves the most relevant chunks with FAISS, and answers the question using OpenRouter when configured.

## Features

- Upload PDF files directly from the Streamlit interface.
- Extract text page by page with `pdfplumber`.
- Preserve source filename and page metadata.
- Split extracted text into overlapping chunks.
- Search PDF chunks locally with FAISS.
- Generate grounded answers through OpenRouter.
- Continue working without an API key using a local extractive-answer fallback.
- Store uploaded PDFs locally in the ignored `pdfs/` directory.

## Project Structure

```text
.
├── app.py              # PDF loading, chunking, and local embeddings
├── frontend.py         # Streamlit user interface
├── rag_pipeline.py     # Retrieval context and OpenRouter answering
├── .env.example        # Safe environment variable template
├── .gitignore          # Ignores secrets, uploads, and virtual environments
└── pdfs/               # Runtime upload directory, created automatically
```

## Requirements

- Python 3.10 or newer
- A virtual environment is recommended
- An OpenRouter API key is optional

## Installation

From the project directory:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On this project, the equivalent virtual-environment commands are:

```bash
venv/bin/python -m pip install streamlit pdfplumber numpy faiss-cpu openai python-dotenv langchain-core langchain-community langchain-text-splitters
```

Or install from the requirements file directly:

```bash
venv/bin/python -m pip install -r requirements.txt
```

## Configuration

Copy the example configuration to `.env`:

```bash
cp .env.example .env
```

Then edit `.env` and add your OpenRouter key:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=openai/gpt-4o-mini
OPENROUTER_SITE_URL=http://localhost:8501
OPENROUTER_APP_NAME=PDF Research Assistant
```

`.env` is ignored by Git. Never commit a real API key. The checked-in `.env.example` must contain only placeholder values.

### Choosing an OpenRouter model

`OPENROUTER_MODEL` must be an OpenRouter model identifier supported by your account. For example:

```env
OPENROUTER_MODEL=openai/gpt-4o-mini
```

The application does not make an OpenRouter request during startup. A request is made only after a PDF is uploaded and a question is submitted.

## Running the App

```bash
venv/bin/streamlit run frontend.py
```

Or, after activating the virtual environment:

```bash
streamlit run frontend.py
```

Open the local URL shown by Streamlit, usually `http://localhost:8501`.

## How It Works

1. The user uploads a PDF.
2. The PDF is saved in the local `pdfs/` directory.
3. `pdfplumber` extracts text from each page.
4. `RecursiveCharacterTextSplitter` creates chunks of about 1,000 characters with overlap.
5. `HashEmbeddings` creates deterministic local vectors without downloading an embedding model.
6. FAISS retrieves the four chunks most similar to the question.
7. If `OPENROUTER_API_KEY` is configured, the retrieved context is sent to the selected OpenRouter model.
8. If OpenRouter is unavailable or not configured, the app returns a local extractive answer from the retrieved text.

## Running Without an API Key

The app still runs without OpenRouter credentials. PDF extraction, embeddings, FAISS search, and extractive answers all run locally.

The local fallback is useful for development, but it is less capable than an LLM. It ranks sentences based on word overlap, so it may struggle with paraphrased questions, multi-step reasoning, tables, scanned PDFs, and images.

## Supported PDFs

The current text extraction works best with digitally generated PDFs that contain selectable text. Scanned or image-only PDFs may produce this error:

```text
The PDF does not contain extractable text
```

OCR support is not included yet.

## Troubleshooting

### `OPENROUTER_API_KEY` is not working

Check that:

- The key is in `.env` in the project directory.
- The variable name is exactly `OPENROUTER_API_KEY`.
- The key has not been surrounded by accidental extra characters.
- The selected model is available through OpenRouter.
- The key has not been committed or exposed publicly.

Restart Streamlit after changing `.env`.

### The app returns a local-style answer

This usually means the OpenRouter key was not loaded, the placeholder value is still present, or the OpenRouter request failed. The application intentionally falls back to a local answer instead of stopping.

### FAISS installation errors

FAISS wheels can depend on the Python version and operating system. Use the project virtual environment and upgrade `pip` before installing dependencies:

```bash
venv/bin/python -m pip install --upgrade pip
venv/bin/python -m pip install faiss-cpu
```

## Validation

Compile the project files:

```bash
venv/bin/python -m py_compile app.py rag_pipeline.py frontend.py
```

Run a local retrieval smoke test:

```bash
OPENROUTER_API_KEY=your_openrouter_api_key_here venv/bin/python - <<'PY'
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from app import create_chunks, get_embeddings_model
from rag_pipeline import answer_query, llm_model

documents = create_chunks([
    Document(page_content="The project deadline is Friday.")
])
store = FAISS.from_documents(documents, get_embeddings_model())
retrieved = store.similarity_search("When is the project deadline?", k=1)
answer = answer_query(retrieved, llm_model, "When is the project deadline?")
assert "Friday" in answer.content
print(answer.content)
PY
```

## Security Notes

- Keep `.env` private.
- Do not paste API keys into source files.
- Do not commit uploaded PDFs if they contain private information.
- Review the PDF context before sending it to a hosted model such as OpenRouter.
- The current app stores uploaded files under `pdfs/` on the local machine.

## License

No license has been specified for this project yet.
