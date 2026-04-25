## Customer Support AI Agent

An intelligent customer support chatbot built with LangChain, 
Pinecone, OpenAI and Streamlit.

## Features
- RAG-powered answers from a knowledge base (Pinecone)
- Automatic support ticket creation (SQLite)
- Human escalation detection
- Full conversation memory
- Live ticket dashboard

## Tech Stack
- LangChain (Agent framework)
- OpenAI GPT-4o-mini (Language model)
- Pinecone (Vector database)
- Streamlit (Web UI)
- SQLite (Ticket database)

## Setup
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Add your API keys to `.env`
4. Ingest documents: `python knowledge_base.py`
