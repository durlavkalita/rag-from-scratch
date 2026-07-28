# Compliance-Aware Search & Response Agent System

## Objective
Build a multi-agent Command Line Interface (CLI) system in pure Python that parses corporate policy guidelines, performs standalone information retrieval using information-retrieval math from scratch, evaluates query compliance risks, and provides structured response architectures.

## Architecture & File Structure
The codebase must be modularly split across four specific files:

```text
├── rag.py               # Document parsing, TF-IDF calculation, and Similarity Search Engine
├── search_agent.py      # Core processing agent wrapping the RAG Engine for metadata & intent
├── response_agent.py    # Formatting engine translating contexts into rule-driven configurations
└── orchestrator.py      # Core CLI executive loop, handling error states and tracking state logs
```

## Component 1 — RAG Engine (rag.py)
Implement a Retrieval-Augmented Generation system from scratch:

- Parse and chunk the policy document into overlapping text chunks
- Build a TF-IDF vocabulary matrix using numpy — no sklearn
- Implement cosine similarity search using numpy to retrieve the top-K most relevant chunks for a query
- Store chunks and their TF-IDF vectors in memory (your "vector store")

## Component 2 — Search Agent (search_agent.py)
A class that wraps the RAG engine and exposes structured search capabilities:

- search(query: str) -> list[dict] — returns top matching chunks with relevance scores
- get_intent(query: str) -> str — classifies the query intent as one of: COMPLIANT, RISKY, RESTRICTED, NEUTRAL — based on keywords and context retrieved from the document
- get_context(query: str) -> str — returns a condensed context string from top chunks


## Component 3 — Response Agent (response_agent.py)
A class that calls the Search Agent and composes a final structured response:

- generate_response(query: str) -> dict — returns a dict with:

  - query — original query
  - intent — from get_intent()
  - context — relevant policy excerpts
  - answer — a rule-based or template-based natural language answer derived from the context
  - confidence — a float score (0–1) based on cosine similarity of top result




## Component 4 — Orchestrator (orchestrator.py)
A controller that manages the full pipeline and exposes it via CLI:

- Accepts commands like:

    - search <query> — runs a search and prints ranked results
    - ask <query> — runs the full response pipeline and prints structured output
    - intent <query> — prints only the intent classification
    - exit — quits
- Handles error cases (empty query, no results found, etc.)
- Logs each interaction to a file (session.log)

## Constraints

- Only numpy and Python standard library (re, math, os, json, collections, etc.)
- No external NLP or ML libraries whatsoever
- No API calls
- Everything runs locally and in-memory (except the log file)
- Code must be split across the files above — no monolith

## Evaluation Criteria

- Correctness of TF-IDF implementation
- Quality of cosine similarity retrieval
- Clean class design and separation of concerns
- Intent classification logic tied to document content
- CLI usability and error handling