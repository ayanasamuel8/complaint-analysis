from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
import os
import pickle
from src.config import PROCESSED_DATA_PATH
from src.utils import load_data

# CONFIGURABLE
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# 2. Chunking narratives
def chunk_narratives(df: pd.DataFrame, text_column: str) -> list:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    all_chunks = []
    metadata = []

    for _, row in df.iterrows():
        chunks = splitter.split_text(row[text_column])
        for chunk in chunks:
            all_chunks.append(chunk)
            metadata.append({
                "complaint_id": row["Complaint ID"],
                "product": row["Product"],
                "text": chunk
            })
    return all_chunks, metadata

# 3. Generate vector embeddings
def embed_chunks(chunks: list, model_name: str):
    model = SentenceTransformer(model_name)
    vectors = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    return vectors

# 4. Create FAISS index
def index_to_faiss(vectors, metadata, index_path="../vector_store/index.faiss", meta_path="../vector_store/metadata.pkl"):
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    faiss.write_index(index, index_path)

    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)

    print(f"✅ Indexed {len(vectors)} vectors. FAISS index saved to {index_path}.")

# 5. Main runner
def run_embedding_pipeline():
    df = load_data(PROCESSED_DATA_PATH)
    print(f"📄 Loaded {len(df)} complaints.")

    print("🔪 Chunking narratives...")
    chunks, metadata = chunk_narratives(df, text_column="Cleaned_Narrative")

    print("🔗 Embedding chunks...")
    vectors = embed_chunks([meta["text"] for meta in metadata], EMBEDDING_MODEL)

    print("📦 Indexing into FAISS...")
    index_to_faiss(vectors, metadata)

if __name__ == "__main__":
    run_embedding_pipeline()
