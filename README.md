# 🏥 Healthcare RAG Q&A Pipeline

An end-to-end **Retrieval-Augmented Generation (RAG)** system for medical document question answering. Built with LangChain, FAISS, HuggingFace embeddings, and Groq LLM — completely free to run.

> Built as part of an AI research internship application focused on Alzheimer's and cancer detection research.

---

## Live Demo

Ask natural language questions about Alzheimer's disease and dementia. The system retrieves relevant chunks from medical documents and generates grounded, trustworthy answers — it will never hallucinate information that isn't in the source material.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.2-green)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![Groq](https://img.shields.io/badge/LLM-Groq%20Llama3-orange)
![FAISS](https://img.shields.io/badge/VectorStore-FAISS-purple)

---

## How It Works
PDF Documents → Chunking → HuggingFace Embeddings → FAISS Vector Store
↓
User Question → Embed Query → Similarity Search → Retrieved Chunks
↓
Groq LLM (Llama 3.1) → Answer
---

## Tech Stack

| Component | Technology |
|---|---|
| Orchestration | LangChain |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | FAISS |
| LLM | Groq API (Llama 3.1 8B Instant) |
| Web UI | Streamlit |
| Document Loader | PyPDF |

---

## Project Structure
rag-healthcare-qa/
├── docs/                  # Source medical PDFs (not committed)
├── vectorstore/           # FAISS index (auto-generated)
├── ingest.py              # Document ingestion pipeline
├── rag_chain.py           # CLI question answering
├── app.py                 # Streamlit web interface
├── evaluate.py            # Pipeline evaluation
└── .env                   # API keys (not committed)
---

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/adeeba26/rag-healthcare-qa.git
cd rag-healthcare-qa
```

### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install langchain langchain-community langchain-groq langchain-core \
            langchain-text-splitters faiss-cpu pypdf python-dotenv \
            sentence-transformers streamlit groq torchvision
```

### 4. Set up API key
Create a `.env` file:
GROQ_API_KEY=your_groq_api_key_here
Get a free key at [console.groq.com](https://console.groq.com)

### 5. Add your documents
Drop PDF files into the `docs/` folder.

### 6. Build the vector store
```bash
python ingest.py
```

### 7. Run the web app
```bash
streamlit run app.py
```

Or use the CLI:
```bash
python rag_chain.py
```

---

## Evaluation Results

```bash
python evaluate.py
```

| Metric | Score |
|---|---|
| Source retrieval rate | 100% |
| Answer rate | 60% |
| Keyword match score | 53% |

The **100% source retrieval rate** confirms the vector store consistently finds relevant document chunks. Scores improve significantly with more documents — after adding a second PDF (146 pages, 1698 chunks), answer quality improved noticeably.

---

## Key Design Decisions

- **Chunk size 500, overlap 100** — balances context preservation with retrieval precision for dense medical text
- **Top-k=4 retrieval** — fetches 4 most relevant chunks per query, providing sufficient context without overwhelming the prompt
- **Temperature=0** — ensures deterministic, factual answers suitable for medical information
- **Grounded prompting** — the system prompt explicitly instructs the LLM to only use retrieved context, preventing hallucination

---

## Future Improvements

- [ ] RAGAS evaluation for faithfulness and answer relevancy scoring
- [ ] Add reranking with cross-encoders for improved retrieval precision
- [ ] Deploy to Streamlit Cloud (public URL)
- [ ] Multi-agent research assistant (Project 2)
- [ ] Fine-tuning embeddings on medical vocabulary

---

## Author

**Adeeba Akhtar** — AI Researcher · LLM & RAG Engineer  
Aligarh Muslim University  
[adeebaakhtar2026@gmail.com](mailto:adeebaakhtar2026@gmail.com)