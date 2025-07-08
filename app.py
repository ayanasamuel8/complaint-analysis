import streamlit as st
import textwrap
import time
from src.evaluation import RAGEvaluator

# --- Page Setup ---
st.set_page_config(page_title="CrediTrust Complaint Chatbot", layout="centered")
st.title("🧠 Intelligent Complaint Analysis")
st.markdown("Ask any question about customer complaints and get a grounded answer with real sources.")

# --- Helper: Wrap Sources Neatly ---
def wrap_sources(sources: str, width: int = 90) -> str:
    source_chunks = sources.strip().split("\n---\n")
    formatted_chunks = []

    for i, chunk in enumerate(source_chunks, start=1):
        lines = chunk.strip().split("\n")
        joined = " ".join(line.strip() for line in lines if line.strip())
        wrapped = textwrap.fill(joined, width=width)
        formatted = f"Source {i}:\n{wrapped}"
        formatted_chunks.append(formatted)

    return "\n\n".join(formatted_chunks)

# --- Session State Initialization ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Evaluator Setup ---
evaluator = RAGEvaluator("vector_store/index.faiss", "vector_store/metadata.pkl")

# --- Sample Questions ---
example_questions = [
    "Why are customers frustrated with credit card charges?",
    "What issues do users report about loan applications?",
    "Are there complaints about Buy Now, Pay Later?",
    "Do customers complain about savings account closures?",
    "How often do users face money transfer failures?",
    "Are there any mentions of late fees?",
    "What are common disputes with customer support?",
    "Do people complain about app crashes or bugs?",
]

# --- Input UI ---
question = st.text_input("💬 Ask your question:", placeholder="Type your question here...")
example = st.selectbox("📌 Or choose an example question:", [""] + example_questions)

if example and not question.strip():
    question = example

col1, col2 = st.columns([1, 1])
ask_clicked = col1.button("🚀 Ask")
clear_clicked = col2.button("🧹 Clear")

# --- Clear Chat ---
if clear_clicked:
    st.session_state.chat_history = []
    st.rerun()

# --- Ask and Add to Chat ---
if ask_clicked and question.strip():
    with st.spinner("Thinking..."):
        df = evaluator.evaluate_questions([question])
        answer = df["Generated Answer"].iloc[0]
        sources = df["Retrieved Sources"].iloc[0]
        st.session_state.chat_history.append({
            "question": question,
            "answer": answer,
            "sources": wrap_sources(sources, width=100)
        })

# --- Display Chat History ---
for entry in reversed(st.session_state.chat_history):  # Most recent first
    with st.container():
        st.markdown(f"**🧾 Question:** {entry['question']}")
        
        st.subheader("✅ Answer")
        # Streaming simulated again for each entry
        placeholder = st.empty()
        streamed = ""
        for token in entry["answer"].split():
            streamed += token + " "
            placeholder.markdown(streamed + "▌")
            time.sleep(0.01)  # Faster for previous messages
        placeholder.markdown(streamed)

        st.subheader("📚 Source Excerpts")
        st.code(entry["sources"])
        st.markdown("---")  # Divider between chats
