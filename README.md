# BatChat RAG Engine (Backend) ⚙️

This is the FastAPI backend powering the BatChat Chrome extension. It handles web scraping, text chunking, local vector embeddings, and LLM orchestration using a strictly grounded RAG prompt.

## 🏗 Architecture & Engineering Highlights

* **Zero-Cost Local Embeddings:** Vectors are computed entirely on local hardware using an embedded, SQLite-backed ChromaDB instance with the default MiniLM ONNX model. No external vector DB or embedding API keys are required.
* **Intent-Driven Routing:** A custom LLM intent router acts as a compute governor. It classifies queries (`SPECIFIC`, `GLOBAL`, `HYBRID`, `CHAT`) to dynamically adjust retrieval breadth ($n$ chunks) or short-circuit retrieval entirely for casual greetings, saving expensive tokens.
* **SSRF-Hardened BYOK Validation:** The API accepts a `provider_id` (e.g., "groq", "openai") from the extension rather than a raw `base_url`. The backend resolves this against a hardcoded allow-list, preventing Server-Side Request Forgery (SSRF) hijacking.
* **Unified Single-Pass Deep Crawl:** For full-site indexing, the system looks for a sitemap. If none is found, it falls back to a custom Breadth-First Search (BFS) via `Crawl4AI` that scrapes markdown and extracts outgoing links in the exact same pass, halving compute overhead.

## 🛠 Tech Stack

* **API Framework:** FastAPI, Uvicorn
* **Vector Database:** ChromaDB (PersistentClient)
* **Crawler:** Crawl4AI (Playwright headless browser)
* **LLM Client:** OpenAI Python SDK (compatible with Groq, OpenAI, OpenRouter)

## 🚀 Getting Started

### Prerequisites
* Python 3.10+
* *(Windows Users)* The `main.py` entry point automatically configures the `WindowsProactorEventLoopPolicy` required by Playwright.

### Installation

1.  Clone the repository and create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
3.  *(Optional)* Install Playwright browsers if using Crawl4AI for the first time:
    ```bash
    playwright install
    ```

### Configuration

Create a `.env` file in the root directory. You can provide an optional server-side fallback key for local testing when the UI doesn't provide one:
```env
GROQ_API_KEY=gsk_your_fallback_key_here