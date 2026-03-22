# 🎬 Movie Recommendation System using RAG (Retrieval-Augmented Generation)

## Overview
This project builds a **Movie Recommendation System** using a **RAG (Retrieval-Augmented Generation)** pipeline. It leverages:
- IMDb Top 250 Movies dataset
- OpenAI Embeddings
- LlamaIndex (Vector Store Index)
- OpenAI LLM for intelligent recommendations

The system retrieves similar movies based on semantic similarity and generates human-like explanations.

---

## Architecture

```
IMDb Dataset
   ↓
Document Creation
   ↓
OpenAI Embeddings (text-embedding-3-small)
   ↓
VectorStoreIndex (LlamaIndex)
   ↓
Similarity Search
   ↓
OpenAI LLM (gpt-4o-mini)
   ↓
Movie Recommendations + Explanation
```

---

## 🚀 Features

- Semantic movie search using embeddings
- Context-aware recommendations
- Explanation of similarity between movies
- Includes metadata like:
  - Director
  - Rating
  - Release Year

---

##Dataset

- IMDb Top 250 Movies dataset (CSV format)
- Fields used:
  - Movie Description
  - Release Year
  - Director
  - Rating

---

## Tech Stack

- Python
- Pandas
- Scikit-learn
- OpenAI API
- LlamaIndex

---

## Installation

```bash
pip install pandas scikit-learn
pip install llama-index llama-index-embeddings-openai llama-index-llms-openai
```

---

## Setup OpenAI API Key

1. Store your API key in a text file:
```
OpenAI_API_Key.txt
```

2. Load it in the notebook:
```python
with open('OpenAI_API_Key.txt', 'r') as f:
    openai.api_key = f.read().strip()
```

3. Set environment variable:
```python
import os
os.environ["OPENAI_API_KEY"] = openai.api_key
```

---

## Data Processing

- Load dataset using Pandas
- Convert each movie into a structured document

```python
from llama_index.core import Document

movie_text = f"""
Movie Title: {row['Description']}
Release Year: {row['Release_Year']}
Director: {row['Director']}
Rating: {row['Rating']}
Description: {row['Description']}
"""
```

---

## Embeddings & Indexing

```python
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

embed_model = OpenAIEmbedding(model="text-embedding-3-small")
llm = OpenAI(model="gpt-4o-mini")
```

```python
from llama_index.core import VectorStoreIndex, Settings

Settings.embed_model = embed_model
Settings.llm = llm

index = VectorStoreIndex.from_documents(documents)
```

---

## Query Engine (RAG)

```python
query_engine = index.as_query_engine(similarity_top_k=5)
```

---

## Example Query

```python
query = """
Recommend movies similar to Interstellar.
Explain why they are similar.
Mention rating and director.
"""

response = query_engine.query(query)
print(response)
```

---

## Output

- Top similar movies
- Explanation of similarity
- Metadata (director, rating, etc.)

---


## 🙌 Conclusion

This project demonstrates how to combine **vector search + LLMs** to build an intelligent recommendation system using RAG architecture.

--

