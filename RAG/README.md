📚 RAG-Based Document Question Answering System

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline using LangChain, OpenAI models, and Chroma vector database to answer questions from a PDF document.

---

##  Overview

The system:

1. Loads a PDF document
2. Splits it into semantic chunks
3. Converts text into embeddings using OpenAI
4. Stores embeddings in Chroma vector database
5. Retrieves relevant chunks for a user query
6. Reranks results using a Cross-Encoder model
7. Uses GPT-4o-mini to generate an answer grounded in context

---

##  Architecture

User Question  
↓  
Vector Search (Top K Results)  
↓  
Cross-Encoder Reranking  
↓  
Top Relevant Chunks  
↓  
LLM (GPT-4o-mini)  
↓  
Final Answer  

---

## Tech Stack

- Python
- LangChain
- OpenAI (ChatOpenAI + OpenAIEmbeddings)
- ChromaDB
- SentenceTransformers (Cross-Encoder)
- PyPDF Loader

---

## Installation

```bash
pip install -U langchain langchain-openai langchain-community
pip install langchain-text-splitters chromadb pypdf
pip install sentence-transformers

Environment Setup

Set your OpenAI API key:

export OPENAI_API_KEY="your_api_key_here"


Project Workflow
1️⃣ Load PDF
loader = PyPDFLoader("Route.pdf")
documents = loader.load()


Split Documents

Create Embeddings & Store in Vector DB

Cross-Encoder Reranking

Build RAG Chain

