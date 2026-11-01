Production-Style RAG Assistant



A grounded Retrieval-Augmented Generation (RAG) system built with LangChain, ChromaDB, HuggingFace Embeddings, and Groq.



Designed with production engineering standards: hybrid retrieval (dense vector + sparse BM25), source citations, strict refusal guardrails against hallucinations and prompt injection, automated evaluation suites, unit testing with `pytest`, Docker containerization, and GitHub Actions CI.



\---



Architecture \& Pipeline



text

\[ Documents (PDF / MD / TXT) ]

&#x20;             │

&#x20;             ▼

&#x20;  \[ Chunking \& Ingestion ] ──► (ChromaDB Vector Store + Local Embeddings)

&#x20;             │

&#x20;             ▼

&#x20;     \[ User Question ]

&#x20;             │

&#x20;             ▼

&#x20;   \[ Hybrid Retriever ]

&#x20;    ├── Dense Search (Chroma Cosine Similarity, weight: 0.6)

&#x20;    └── Sparse Search (BM25 Keyword Matching, weight: 0.4)

&#x20;             │

&#x20;             ▼

&#x20; \[ Context Formatting with \[n] Citations ]

&#x20;             │

&#x20;             ▼

&#x20; \[ Grounded LLM Generation (Groq) ] ──► (Strict Refusal if Context Missing)



\---



Key Features



\- Hybrid Retrieval: Combines semantic vector search (`Chroma` + `sentence-transformers/all-MiniLM-L6-v2`) with lexical keyword search (`BM25Retriever`) via an `EnsembleRetriever` to capture both conceptual meaning and exact policy terms.

\- In-Memory Embedding Caching: Uses `@lru\_cache` on the embedding initialization layer so model weights load once into memory rather than re-instantiating per query.

\- Strict Grounding \& Citations: Enforces source attribution (`\[1]`, `\[2]`) and deterministic refusal (`"I don't have enough information in the provided documents to answer that."`) when queried outside the knowledge base or targeted by prompt injection.

\- Automated Evaluation Suite (`evals/`): Validates grounded accuracy and refusal behavior against a curated `golden\_set.jsonl` benchmark.

\- CI/CD \& Unit Testing (`tests/`): Runs automated `pytest` checks on citation formatting, guardrail prompts, and dataset schemas via GitHub Actions on every push.



\---



Tech Stack



\- Framework: LangChain (`langchain-core`, `langchain-community`, `langchain-chroma`, `langchain-groq`)

\- Vector Database \& Lexical Search: ChromaDB + BM25 (`rank\_bm25`)

\- Embeddings: HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)

\- LLM Engine: Groq API (`qwen/qwen3.8-27b`)

\- Configuration: Pydantic v2 (`pydantic-settings`)

\- UI \& Deployment: Streamlit, Docker

\- Testing \& CI: Pytest, GitHub Actions



\---



&#x20;Getting Started



&#x20;1. Setup Virtual Environment



powershell

python -m venv .venv

.venv\\Scripts\\activate

pip install -r requirements.txt





2\. Configure Environment Variables



Create a `.env` file in the project root (or copy `.env.example`):



env

GROQ\_API\_KEY=your\_groq\_api\_key\_here

GROQ\_CHAT\_MODEL=qwen/qwen3.8-27b

CHUNK\_SIZE=600

CHUNK\_OVERLAP=100

TOP\_K=8





&#x20;3. Ingest Documents \& Run the App



Place PDF, Markdown, or text files inside `data/`, then run:



powershell

python -m src.ingest

streamlit run src/app.py

```



\---



&#x20;Running Evaluations \& Tests



&#x20;Run Unit Tests (Pytest)



powershell

python -m pytest -v





Run End-to-End Golden Set Evaluation



powershell

python evals/run\_eval.py





\---



Key Engineering Takeaways



\- Hybrid Search Outperforms Pure Vector Search:Combining dense embeddings (0.6 weight) with BM25 keyword matching (0.4 weight) prevents missed matches on exact figures, acronyms, and policy numbers.

\- Singleton Model Caching: Caching the `HuggingFaceEmbeddings` instance with `@lru\_cache(maxsize=1)` eliminated redundant weight loading across multi-query evaluation loops.

\- Deterministic Guardrails \& Evaluation:\*\* Testing both positive factual recall and negative out-of-domain/injection refusals in `golden\_set.jsonl` ensures the system fails safely instead of hallucinating.

\- CI-Safe Configuration: Designing Pydantic `Settings` with safe fallbacks allows unit test suites to execute in GitHub Actions without exposing production API secrets.

