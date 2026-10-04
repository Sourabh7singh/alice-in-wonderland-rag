# 📚 Alice in Wonderland RAG Pipeline

A high-precision, performant Retrieval-Augmented Generation (RAG) system built over *Alice's Adventures in Wonderland*. 

This pipeline leverages **LangChain**, **ChromaDB**, **HuggingFace BGE Embeddings**, and **Google Gemini** (`google-genai` SDK) to deliver accurate, fact-grounded responses, handle false-premise "trick" questions gracefully, and auto-manage local database persistence.

---

## 🛠️ Key Implementation Details

- **SDK & LLM Engine:** Google GenAI SDK (`google.genai.Client`) using `gemini-2.5-flash`.
- **Environment Management:** Uses `python-dotenv` to securely load `GEMINI_API_KEY` from a `.env` file.
- **Dynamic DB Naming & Persistence:** Automatically derives the database directory name from the source file path (e.g., `data/alice_in_wonderland.md` ➔ `alice_in_wonderland_db`).
- **Auto-Initialization:** The `Rag` class automatically checks if the database directory exists. If missing, it loads data, chunks text, and builds the Chroma store before querying.
- **Document Chunking:**
  - `MarkdownHeaderTextSplitter` splits on `("CHAPTER", "Chapter")` headers while keeping structural headers intact (`strip_headers=False`).
  - `RecursiveCharacterTextSplitter` chunks sections into 800-character segments with a 100-character overlap.
- **Retrieval Engine:**
  - **Embedding Model:** `BAAI/bge-base-en-v1.5` with normalized embeddings running on CPU.
  - **BGE Query Prefix:** Prepends `Represent this sentence for searching relevant passages:` to maximize semantic search precision.
  - **Search Strategy:** Uses **Maximal Marginal Relevance (MMR)** (`search_type="mmr"`, `k=4`, `lambda_mult=0.7`) to balance context relevance and chunk diversity.

---

## 📂 Project Structure

```text
alice-in-wonderland-rag/
├── .env                    # Stores GEMINI_API_KEY
├── .gitignore
├── README.md
├── requirements.txt
├── rag.py                  # Complete Rag class implementation & test script
└── data/
    └── alice_in_wonderland.md  # Source markdown text

```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone [https://github.com/Sourabh7singh/alice-in-wonderland-rag.git](https://github.com/Sourabh7singh/alice-in-wonderland-rag.git)
cd alice-in-wonderland-rag

```

### 2. Set Up Virtual Environment & Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

```

### 3. Configure API Key

Create a `.env` file in the root directory and add your Gemini API key:

```env
GEMINI_API_KEY=your_gemini_api_key_here

```

---

## 🧪 Quick Start & Testing

To run a test query through the pipeline:

```bash
python rag.py

```

### Code Usage Example

```python
from rag import Rag

# Initialize RAG (automatically creates/loads database directory)
rag = Rag(file_path="data/alice_in_wonderland.md")

# Query the pipeline
question = "Why was the Cheshire Cat put in prison by the Queen of Hearts?"
response = rag.ask_query(question)

print(response)

```