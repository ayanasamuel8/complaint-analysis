# 🧠 Intelligent Complaint Analysis for Financial Services

This project builds a RAG-based chatbot to help internal teams at CrediTrust Financial understand customer complaints quickly and accurately. The system allows users to ask plain-English questions and receive grounded answers powered by real complaint narratives.

---

## 💼 Business Context

CrediTrust Financial is a fast-growing digital finance company operating in East Africa. It offers:

* 💳 Credit Cards
* 🧾 Personal Loans
* 🛒 Buy Now, Pay Later (BNPL)
* 🏦 Savings Accounts
* 💸 Money Transfers

Thousands of complaints are submitted monthly. Today, product and support teams spend hours manually reviewing this data. This tool will automate that process using AI.

---

## 🚀 Key Features

* 🔍 Semantic Search on real complaint narratives
* 🤖 LLM-generated insights with evidence
* 🧱 Modular pipeline (preprocessing → chunking → embedding → RAG)
* 💬 User-friendly chatbot interface (Gradio or Streamlit)

---

## 📁 Project Structure

```
project-root/
├── README                        👈 This file
├── requirements.txt             📦 Python dependencies
├── data/
│   ├── raw/
│   │   └── complaints.csv       🗃️ Raw CFPB complaint data
│   └── processed/
│       └── processed_complaints.csv  ✅ Cleaned and filtered data
├── notebooks/
│   └── 1.0-eda.ipynb            📊 EDA and preprocessing
├── report/                      📸 Visualizations, screenshots
├── src/
│   ├── __init__.py
│   ├── config.py                ⚙️ Configuration and constants
│   ├── data_processing.py       🧹 Cleaning and filtering logic
│   └── utils.py                 🔧 Helper functions
├── tests/
│   ├── __init__.py
│   └── test_dummy.py            🧪 Unit test scaffolding
```

---

## ⚙️ Setup Instructions

1. 🧬 Create a virtual environment
2. 📥 Install dependencies using `pip install -r requirements.txt`
3. 🗂️ Place the CFPB dataset at `data/raw/complaints.csv`
4. 📊 Run the EDA notebook to generate the cleaned dataset
5. 🧠 Continue with chunking, embedding, and chatbot development

---

## 🛠️ Workflow Steps

✅ Task 1: Data cleaning and filtering
✅ Task 2: Chunking narratives + generating embeddings
✅ Task 3: Retrieval + LLM-powered generation
✅ Task 4: Interactive app for internal users

---

## 📦 Dependencies

* `pandas`, `numpy`, `matplotlib`, `seaborn`
* `sentence-transformers`, `faiss-cpu` or `chromadb`
* `langchain`, `transformers`, `openai`, etc.
* `gradio` or `streamlit` for the UI

---

## 🧪 Testing

Use `pytest` to run test files in the `tests/` directory.

---

## 👥 Team

Facilitator: Mahlet
Team: Kerod, Rediet, Rehmet

---

## 🔮 Future Improvements

* ✨ Better prompt engineering
* 🌍 Multilingual support
* 🔁 Real-time complaint ingestion
* ⚙️ CI/CD automation pipeline

---
