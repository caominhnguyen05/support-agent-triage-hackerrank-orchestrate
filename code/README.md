# Multi-Domain Support Triage Agent

A terminal-based multi-domain support triage agent for the HackerRank Orchestrate hackathon.

Processes support tickets across **HackerRank**, **Claude**, and **Visa** domains using a Retrieval-Augmented Generation (RAG) pipeline with semantic search over a ChromaDB vector index and OpenAI GPT-4 for reasoning and response generation.

---

## Key Features

- Multi-domain ticket triage (HackerRank, Claude, Visa)
- Retrieval-Augmented Generation (RAG) with ChromaDB
- Automatic escalation for unsafe or low-confidence queries
- Structured JSON response generation using LLMs
- Token-aware chunking for improved retrieval accuracy

---

## Architecture

```
support_tickets.csv
       │
       ▼
  [main.py]  ← entry point
       │
       ▼
  [agent.py]  ← 5-step pipeline
    │
    ├── Step 1: Domain classification (hackerrank / claude / visa / general)
    ├── Step 2: Vector retrieval from ChromaDB
    ├── Step 3: Safety / escalation gate
    ├── Step 4: OpenAI LLM response generation (using information from the corpus)
    └── Step 5: Structured JSON output parsing
       │
       ▼
  output.csv
```

---

## Approach Overview

This project implements a retrieval-augmented support triage agent that classifies, answers, or escalates support tickets using domain-specific documentation from the provided support corpus.

### 1. Data Preparation (`scraper.py`)

Markdown documentation is converted into clean, structured `.txt` files for embeddings creation:

- Removes YAML frontmatter and Markdown formatting
- Standardizes metadata (SOURCE, TITLE, DOMAIN)
- Flattens directory structure for easier indexing

Why: Clean, normalized text improves embedding quality and retrieval accuracy.

### 2. Vector Indexing (`build_index.py`)

- The processed text from `.txt` files are split into token-based overlapping chunks (using OpenAI's Python Library `tiktoken`)
- Vector embeddings are created for each chunk using `text-embedding-3-small` from OpenAPI
- These embeddings are stored in ChromaDB as a persistent vector database for efficient semantic retrieval

Why these choices:

- OpenAI `text-embedding-3-small` to create embeddings: this model provides strong semantic retrieval performance at a significantly low cost. Offers higher efficiency than previous models like `text-embedding-ada-002`, and much cheaper than `text-embedding-3-large`.
- `tiktoken` for token-aware chunking: ensures text is split based on the same tokenization scheme used by OpenAI models, rather than naive character or word counts.
- ChromaDB for embedding storage: a lightweight, local, and easy-to-use vector database for fast similarity search and persistence

### 3. Triage Pipeline (`agent.py`)

Each ticket goes through a 5-step pipeline:

#### 3.1. Domain Classification

- Rule-based keyword matching to route queries (HackerRank, Claude, Visa)

#### 3.2. Vector Retrieval

- Top-K relevant chunks fetched from ChromaDB using semantic similarity

#### 3.3. Escalation Decision

Tickets are automatically escalated when:

- Sensitive keywords are detected (e.g., fraud, unauthorized access, legal, billing disputes, account lockout)
- Retrieval confidence is too low — specifically, when the maximum similarity score across retrieved chunks is below 0.4.
  This indicates that no sufficiently relevant documentation was found in the corpus, so the agent cannot provide a grounded answer.
- The LLM determines the issue requires human verification

#### 3.4. Response Generation

Uses OpenAI `gpt-4o-mini` with strict grounding rules:

- Must rely only on retrieved context
- Outputs structured JSON
- Formatting requirements (e.g., answer cannot contain markdown fences and symbols, write in paragraphs, use numbered steps if possible)

#### 3.5. Post-processing

- Ensures consistent formatting and safe fallback on parsing errors

Why this approach:

- Retrieval-Augmented Generation (RAG) reduces hallucination
- Deterministic checks (thresholds + keywords) improve reliability
- Structured JSON output simplifies downstream evaluation

### 4. Execution (`main.py`)

- Processes tickets from `support_tickets.csv`
- Runs the triage pipeline per row
- Outputs structured results (`status`, `response`, `request_type`, etc.) to `output.csv`

---

## Install Dependencies and Run the Agent

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

Create a `.env` file in the project root and add your OpenAI API key:

```bash
OPENAI_API_KEY=your_openai_api_key
```

### 3. Scrape the support corpus

```bash
python scraper.py
# Converts all .md files into .txt files and saves articles to data/
```

### 4. Build the vector index

```bash
python build_index.py
# Chunks the articles, create vector embeddings, and stores everything in ChromaDB
```

### 5. Run the agent

Full run on `support_tickets.csv`:

```bash
python main.py
```

Quick test (first 10 rows):

```bash
python main.py --limit 10 --verbose
```

---

## Output Format

Results are written to `support_tickets/output.csv` with these columns:

| Column          | Description                                             |
| --------------- | ------------------------------------------------------- |
| `issue`         | Original ticket text                                    |
| `subject`       | Original subject                                        |
| `company`       | Original company field                                  |
| `status`        | `replied` or `escalated`                                |
| `product_area`  | Inferred support category                               |
| `response`      | Support answer for user or escalation message           |
| `justification` | Agent's reasoning for the decision                      |
| `request_type`  | `product_issue`, `feature_request`, `bug`, or `invalid` |

---

## Project Structure

| File               | Purpose                                                          |
| ------------------ | ---------------------------------------------------------------- |
| `scraper.py`       | Crawls 3 support sites in `data/` folder, saves articles as .txt |
| `build_index.py`   | Chunks + embeds articles into ChromaDB                           |
| `agent.py`         | Core 5-step triage agent pipeline                                |
| `main.py`          | Main entry point to run the agent, handle CSV input/output       |
| `requirements.txt` | Python dependencies used in the project                          |
