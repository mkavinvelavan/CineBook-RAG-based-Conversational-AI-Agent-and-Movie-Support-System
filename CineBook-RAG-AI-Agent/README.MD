# 🎬 CineBook – RAG-based Conversational AI Agent and Movie Support System

CineBook is a Retrieval-Augmented Generation (RAG) powered conversational AI agent designed to deliver intelligent and policy-grounded customer support for a movie ticket booking platform.  
It integrates structured data lookup (movies, theaters, showtimes) with semantic reasoning over policy documents, enabling accurate and human-like responses.

---

## 🚀 Features

- **Hybrid AI System**  
  Combines rule-based lookup tables with Retrieval-Augmented Generation for intelligent query routing.

- **RAG Pipeline Integration**  
  Uses ChromaDB vector database + SentenceTransformer (MiniLM-L6-v2) embeddings for semantic search.

- **LLM-Powered Responses**  
  Utilizes Groq Llama 3.1 8B for contextual, grounded, and policy-aware response generation.

- **Streamlit Chat UI**  
  Real-time conversational interface with session-based context tracking.

- **Structured + Unstructured Data Handling**  
  JSON datasets for movies/theaters/shows + policy documents for refunds, privacy, and user guidelines.

---

## 🧠 Architecture Overview

1. **User Query → Streamlit UI**  
2. **Intent Detection → AI Agent**  
3. **Structured Query?**  
   → Lookup table (movies/theaters/shows)  
4. **Unstructured Query?**  
   → RAG pipeline (embedding → similarity search → context retrieval)  
5. **LLM Reasoning** through Groq Llama 3.1  
6. **Final Human-like Response** displayed in UI

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit**
- **ChromaDB** (vector database)
- **SentenceTransformer MiniLM-L6-v2**
- **Groq Llama 3.1 8B**
- **JSON Data Files**
- **Virtual Environment (venv)**
- **python-dotenv**

---

## 📂 Project Structure

```
CineBook-RAG-AI-Agent/
│
├── app.py
├── requirements.txt
├── .env.example
│
├── data/
│   ├── movies.json
│   ├── theaters.json
│   ├── shows.json
│   ├── refund_policy.json
│   ├── account_privacy_policy.json
│   ├── user_guidelines.json
│
├── README.md
└── document

```

---

## 🔧 Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/cinebook-rag-ai-agent.git
cd cinebook-rag-ai-agent
```

### 2. Create and activate virtual environment
```bash
python -m venv venv
venv\Scripts\activate       # Windows
# or
source venv/bin/activate    # macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Setup

Create a `.env` file in the root directory:

```
GROQ_API_KEY=your_groq_key_here
```

(Do **not** upload your real API key to GitHub. Use `.env.example` to guide users.)

---

## ▶️ Running the Application

```bash
streamlit run app.py
```

This will launch the CineBook chat interface in your browser.

---

## 📸 Screenshots

Add your screenshots inside `/screenshots` folder and reference them here:

```
![Chat UI](screenshots/ui_chat.png)
![Architecture](screenshots/architecture.png)
![Flowchart](screenshots/flowchart.png)
```

---

## 🤖 How It Works

- Structured queries (movies, shows, theaters) → handled via lookup tables  
- Policy queries → handled via RAG pipeline  
- Query embeddings generated using **SentenceTransformer**  
- **ChromaDB** retrieves the top-k relevant policy data  
- **Groq Llama 3.1** generates coherent and policy-aligned responses  
- Streamlit maintains multi-turn chat history  

---

## 📦 Requirements

All dependencies are listed in `requirements.txt`.  
Generate it using:
```bash
pip freeze > requirements.txt
```

---

## 🚀 Future Enhancements

- User authentication  
- Voice-based conversational assistant  
- Integration with a real backend database  
- Admin panel for updating movies/theaters dynamically  

---

## 📄 License
This project is licensed under the **MIT License**.

---

## ⭐ Acknowledgements

- Groq Llama 3.1  
- ChromaDB  
- SentenceTransformer  
- Streamlit

---

## 📬 Contact
For questions or collaboration:  
**Email:** mkavinvelavan@outlook.com  
**LinkedIn:** https://linkedin.com/in/m-kavinvelavan  
