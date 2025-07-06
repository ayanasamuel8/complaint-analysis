# 🧠 Intelligent Complaint Analysis for Financial Services

This project builds a Retrieval-Augmented Generation (RAG) chatbot for CrediTrust Financial to help internal teams quickly analyze and act upon customer complaints using real narratives. Users can ask natural questions and get grounded, AI-powered answers.

---

## 💼 Business Context

CrediTrust Financial is a fast-growing fintech company in East Africa offering:
💳 Credit Cards
🧾 Personal Loans
🛒 Buy Now, Pay Later (BNPL)
🏦 Savings Accounts
💸 Money Transfers

Thousands of complaints are submitted monthly. Currently, product and support teams manually review these complaints — a time-consuming process. This project uses AI to automate insights and enable proactive issue resolution.

---

## 🚀 Key Features

* 🔍 Semantic search over real complaint narratives
* 🤖 LLM-based question answering with source grounding
* 🧱 Modular architecture (EDA → Chunking → Embedding → Retrieval)
* 💬 Interactive chatbot interface for non-technical users

---

## 📁 Project Structure

```
project-root/
├── README 👈 Project overview
├── requirements.txt 📦 Python dependencies
├── data/
│   ├── raw/
│   │   └── complaints.csv 🗃️ Raw CFPB complaint data
│   └── processed/
│       └── processed_complaints.csv ✅ Cleaned and filtered complaints
├── notebooks/
│   └── 1.0-eda.ipynb 📊 EDA and data cleaning
├── report/ 📄 Evaluation, screenshots, final report
├── src/
│   ├── init.py
│   ├── config.py ⚙️ Config and paths
│   ├── data_processing.py 🧹 Text filtering and preprocessing
│   ├── utils.py 🔧 Common functions
│   └── embedding_pipeline.py 🧠 Embedding and FAISS vector indexing
├── tests/
│   ├── init.py
|   └── test_dummy.py 
│   └── test_embedding_pipeline.py 🧪 Unit tests (Pytest)
```

---

## ⚙️ Setup Instructions

* 🧬 Create and activate a virtual environment
* 📥 Install dependencies: `pip install -r requirements.txt`
* 🗂️ Add the `complaints.csv` dataset to `data/raw/`
* 📊 Run the EDA notebook to clean and filter the data
* 🧠 Run `embedding_pipeline.py` to generate and store embeddings
* 💬 Start building the chatbot interface (Gradio/Streamlit)

---

## 🛠️ Workflow Status

* ✅ Task 1: EDA and preprocessing
* ✅ Task 2: Narrative chunking and embedding using FAISS
* ⬜ Task 3: Retrieval + LLM integration with prompt engineering
* ⬜ Task 4: Chatbot UI with source-verified answers

---

## 📦 Dependencies

* pandas, numpy, matplotlib, seaborn
* sentence-transformers, faiss-cpu, langchain
* transformers, openai (optional), gradio or streamlit
* pytest for testing

---

## 🧪 Testing

Run pytest to execute all unit tests:

```
pytest tests/
```

Covers chunking logic, embedding correctness, and FAISS index creation.

---

## 👥 Team

* Facilitator: Mahlet
* Team: Kerod, Rediet, Rehmet

---

## 🔮 Future Improvements

* ✨ Better prompt templates for answer precision
* 🌍 Multilingual complaint support
* 🔁 Real-time ingestion and vector updates
* ⚙️ CI/CD setup with GitHub Actions
* 🧾 Dashboard-style UI and advanced filtering

---

If you want me to convert this to a particular format (e.g., plain text, HTML, or markdown), just let me know!
