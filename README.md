# Retrieval-Augmented Generation (RAG) Pipeline for Question Answering Assistant for Rare Disease Spinocerebellar Ataxia

This repository implements a modular Retrieval-Augmented Generation (RAG) system designed for answering domain-specific questions using scientific PDF documents, with a focus on rare diseases like **Spinocerebellar Ataxia (SCA)**.

---

## 📁 Project Structure
<pre> 
a03/
├── hu_sp25_691_a03-3/ # Main pipeline and source code
│ ├── classes/ # Core pipeline components
│ │ ├── chromadb_retriever.py # Vector search logic
│ │ ├── config_manager.py # Loads config from JSON
│ │ ├── document_ingestor.py # Extracts text from PDFs
│ │ ├── embedding_loader.py # Loads stored embeddings
│ │ ├── embedding_preparer.py # Generates embeddings
│ │ ├── llm_client.py # Sends prompts to local LLM
│ │ ├── rag_query_processor.py # Full RAG orchestration
│ │ └── utilities.py # Common helpers
│ │
│ ├── data/ # Input/output storage
│ │ ├── original_pdf_documents/ # Source PDFs
│ │ ├── raw_input/ # Filenames to ingest
│ │ ├── cleaned_text/ # Cleaned + chunked text files
│ │ ├── embeddings/ # Chunk embeddings in JSON
│ │ ├── vectordb/ # ChromaDB local DB
│ │
│ ├── scripts/ # Bash automation scripts
│ ├── tests/ # Unit tests
│ ├── main.py # Entrypoint for pipeline steps
│ ├── llm_server.py # Starts local LLM server
│ ├── config.json # Main config file
│ └── requirements.txt # Python dependencies
│
├── models/ # Local LLM model files
│ └── mistral-7b-instruct-v0.1.*.gguf
</pre>
---

## ⚙️ Setup Instructions

1. **Clone the repo**:
<pre> 
```bash
git clone https://github.com/your-username/hu_sp25_691_a03-3.git
cd hu_sp25_691_a03-3
</pre>

2. Create virtual environment:
<pre> 
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
</pre>

3. Download LLM locally (e.g., Mistral 7B GGUF)
4. Start the LLM API:
<pre> 
python llm_server.py
</pre>
5. Run the pipeline:
<pre> 
./scripts/run_pipeline_steps.sh
</pre>

🧪 Example Query

❓ "What are some biomarkers of spinocerebellar ataxia?"
The pipeline:

Extracts text from PDFs
Chunks & embeds them
Retrieves top relevant chunks
Feeds them into Mistral-7B for grounded response
✅ Features

Modular & testable class-based structure
Local LLM integration via API
Uses SentenceTransformers for embedding
Retrieval via ChromaDB vector store
Supports RAG and non-RAG response modes
