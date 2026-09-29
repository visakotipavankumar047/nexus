# NEXUS Architecture Overview

## Purpose

NEXUS is an agentic retrieval-augmented generation (RAG) platform. It answers questions by combining
private documents stored in a vector database with live information from the web, and it always
reports which sources an answer came from.

## Request Flow

Every request enters through the FastAPI backend under the `/api/v1` prefix. Route handlers contain no
business logic: they call a service, and the chat service drives the agent. The agent decides which
tools to call, executes them, merges the evidence and generates the final answer.

## Tools

The agent has four tools:

- **vector_search** searches the user's private documents in Pinecone.
- **document_search** searches only inside specific documents chosen by id.
- **web_search** retrieves current information from the internet.
- **calculator** performs exact arithmetic, so the language model never does complex math itself.

The agent may call tools for at most three rounds per answer. This hard limit prevents infinite loops.

## Storage

Pinecone stores document embeddings. Each user has a separate namespace named `user:{user_id}`, so one
user's private vectors can never be returned to another user.

Supabase PostgreSQL stores application state in seven tables: users, conversations, messages, documents,
agent_runs, tool_calls and evaluations.

## Authentication Architecture

Authentication is planned for Phase 8. Until then every request runs as an anonymous user, and all
per-user queries filter on a null user id. When authentication lands, the user id will come from a
verified token and the conversations.user_id column will become NOT NULL.

## Embeddings and Chunking

Documents are split into chunks of about 1000 tokens with 150 tokens of overlap. Each chunk is embedded
with the BAAI/bge-large-en-v1.5 model, which produces 1024-dimensional vectors. The same embedding model
must be used for indexing and for querying; changing it requires re-indexing every document.

## Security

Retrieved documents are treated as untrusted data, never as instructions. The instruction priority is:
system instructions, then developer rules, then the user request, and finally retrieved content.
API keys are never logged, and stack traces are never returned to clients.
