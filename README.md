
# Production-Style RAG Assistant

A grounded Retrieval-Augmented Generation (RAG) app for chatting with documents.

Built as a junior GenAI portfolio project: hybrid retrieval, source citations, and refusal when the answer is not in the documents.

## What it does

- Loads PDF / Markdown / text files from data/
- Splits them into chunks and stores embeddings locally (Chroma + MiniLM)
- Retrieves with hybrid search (vector + BM25)
- Answers with Groq, using only retrieved context
- Cites sources
- Says it does not know when the documents do not contain the answer

## Tech stack

- Python
- LangChain
- Chroma
- sentence-transformers (all-MiniLM-L6-v2)
- Groq (openai/gpt-oss-20b)
- Streamlit

## How to run

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

Create a .env file:

GROQ_API_KEY=your_key_here
GROQ_CHAT_MODEL=openai/gpt-oss-20b
CHUNK_SIZE=600
CHUNK_OVERLAP=100
TOP_K=8

Put documents in data/, then:

python -m src.ingest
streamlit run src/app.py

## Example questions

- What is the data retention period?
- How many remote days are allowed?
- What is the capital of Japan? (should refuse)

## What I learned

- Environment variables override code defaults
- The embedding model used at search time must match ingestion
- LangChain import paths change between versions
- A good RAG system must refuse when context is missing