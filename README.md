# MediAssist 🩺

A RAG-powered medical symptom analysis assistant built with LangChain, ChromaDB, HuggingFace, and FastAPI. Describe your symptoms in plain language and get structured differential diagnoses grounded in a curated knowledge base of 141 diseases — with severity classification, urgency triage, and recommended next steps.

---

## How it works

```
User describes symptoms
        ↓
HuggingFace Embeddings (all-MiniLM-L6-v2)
        ↓
Semantic similarity search over ChromaDB (141 diseases)
        ↓
Top 3 matching disease documents retrieved
        ↓
LangChain LCEL chain (RunnableParallel → Prompt → LLM → Parser)
        ↓
Llama-3.1-8B-Instruct generates structured response
        ↓
FastAPI POST /ask  or  Streamlit UI
```

---

## Features

- **141 diseases** across 19 medical categories (Respiratory, Neurological, Infectious, Oncological, Mental Health, and more)
- **Semantic retrieval** — understands natural language symptoms, not just keyword matching
- **Structured output** — every response includes possible conditions, most likely diagnosis, urgency level, and next steps
- **Source grounding** — answers are based strictly on the knowledge base, not LLM hallucination
- **Dual interface** — REST API via FastAPI + chat UI via Streamlit
- **Input validation** — filters out non-medical inputs before invoking the chain

---

## Project structure

```
mediassist/
├── rag_chain.py      # Core RAG pipeline — ingestion, retrieval, LangChain chain
├── main.py           # FastAPI backend — REST API endpoints
├── app.py            # Streamlit frontend — chat interface
├── data.csv          # 141-disease knowledge base (6 columns, 19 categories)
├── requirements.txt  # Dependencies
├── .env              # HuggingFace API token (not committed)
└── chroma_db/        # Persisted ChromaDB vector store (auto-generated)
```

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/mediassist
cd mediassist
```

### 2. Create virtual environment
```bash
python -m venv myenv
myenv\Scripts\activate        # Windows
source myenv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your HuggingFace token
Create a `.env` file in the project root:
```
HUGGINGFACEHUB_API_TOKEN=your_token_here
```
Get your free token at: https://huggingface.co/settings/tokens

Accept model terms at: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

---

## Running the project

### Option 1 — Terminal (test mode)
Builds ChromaDB and runs an interactive chat loop directly in the terminal:
```bash
python rag_chain.py
```

### Option 2 — FastAPI REST API
```bash
uvicorn main:app --reload
```
Then open `http://127.0.0.1:8000/docs` to test the API interactively.

**POST /ask**
```json
{
  "question": "I have high fever, joint pain and rash for 3 days"
}
```

**Response:**
```json
{
  "answer": "**Possible Conditions:**\n- **Dengue** ..."
}
```

### Option 3 — Streamlit UI
```bash
streamlit run app.py
```
Opens a chat interface at `http://localhost:8501`

---

## Example queries

| Query | Top retrieved condition |
|---|---|
| "High fever, pain behind eyes, joint pain, rash" | Dengue |
| "Severe headache, stiff neck, photophobia, fever" | Meningitis |
| "Fatigue, weight gain, cold intolerance, depression" | Hypothyroidism |
| "Diarrhea, high fever, weakness, headache, vomiting" | Gastroenteritis / Typhoid |
| "Chest pain, shortness of breath, cold sweat" | Heart Attack |

---

## Tech stack

| Component | Technology |
|---|---|
| Document loading | LangChain CSVLoader |
| Text splitting | RecursiveCharacterTextSplitter |
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| Vector store | ChromaDB |
| LLM | Llama-3.1-8B-Instruct (HuggingFace) |
| Chain orchestration | LangChain LCEL |
| API | FastAPI + Uvicorn |
| UI | Streamlit |
| Environment | python-dotenv |

---

## Knowledge base

`data.csv` contains 141 diseases with 6 columns:

| Column | Description |
|---|---|
| Disease | Disease name |
| Category | Medical category (19 total) |
| Severity | Mild / Moderate / Severe classification |
| Symptoms | Primary symptoms (5-8 per disease) |
| Additional_Symptoms | Secondary symptoms to watch for |
| Risk_Factors | Known risk factors for the disease |

---

## Disclaimer

> ⚠️ MediAssist is for **educational and informational purposes only**. It does not constitute medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional for proper evaluation and treatment.

---

## Author

**Dhwanit Chokshi**
- GitHub: [github.com/yourusername](https://github.com/yourusername)
- LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)
- Portfolio: [yourportfolio.com](https://yourportfolio.com)
