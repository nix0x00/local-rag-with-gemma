# Intro to LangChain — Local RAG Pipeline

A minimal Retrieval-Augmented Generation (RAG) pipeline built with LangChain, ChromaDB, and Ollama.

## What it does

[main.ipynb](main.ipynb) walks through building a RAG system end-to-end using  State of the Union address as the knowledge base:

1. **Load & chunk** — reads `state_of_the_union.txt` and splits it into 512-character chunks (50-char overlap) using `CharacterTextSplitter`.
2. **Embed** — generates vector embeddings for each chunk using the `nomic-embed-text` model via Ollama.
3. **Store** — persists embeddings into a ChromaDB collection (`test_collection`) running as a local HTTP server on port 8000.
4. **Retrieve** — performs similarity search against the vector store to pull relevant chunks for a given query.
5. **Generate** — passes retrieved context + query into a `gemma4:e4b` LLM (via Ollama) using a strict prompt that instructs the model to only answer from the provided context.
6. **Chain** — wires everything together as a LangChain LCEL chain: `retriever → formatter → prompt → LLM → output parser`.

The final cell demos the grounding behaviour: questions answerable from the speech get answered; questions outside the document (e.g. "What is RAG?") return "I don't know."

## Stack

| Component | Tool |
|---|---|
| Orchestration | LangChain (LCEL) |
| Embeddings | `nomic-embed-text` via Ollama |
| Vector store | ChromaDB (HTTP client) |
| LLM | `gemma4:e4b` via Ollama |
| Document | State of the Union address |

## Prerequisites

- [Ollama](https://ollama.com) running locally with `nomic-embed-text` and `gemma4:e4b` pulled
- ChromaDB server running on `localhost:8000`
- Python dependencies: `langchain`, `langchain-chroma`, `langchain-ollama`, `chromadb`, `python-dotenv`

```bash
# Pull required models
ollama pull nomic-embed-text
ollama pull gemma4:e4b

# Start ChromaDB
chroma run --port 8000

# Install Python deps
pip install langchain langchain-chroma langchain-ollama chromadb python-dotenv
```

Then open and run [main.ipynb](main.ipynb) top to bottom.

## Notes

- OpenAI embeddings (`text-embedding-3-large`) are stubbed out in comments if you want to swap to a cloud provider — just set `OPENAI_API_KEY` in a `.env` file and uncomment those lines.
- Re-running the embedding cell without clearing the collection will add duplicate documents. Clear the collection first or use a fresh collection name.
