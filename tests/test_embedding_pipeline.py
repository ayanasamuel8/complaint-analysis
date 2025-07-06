import os
import pickle
import pandas as pd
import numpy as np
import faiss
import pytest
import sys

from src.embedding_pipeline import chunk_narratives, embed_chunks, index_to_faiss

# Fixture sample DataFrame
sample_df = pd.DataFrame({
    "Complaint ID": [101, 102],
    "Product": ["Credit card", "Money transfers"],
    "Cleaned_Narrative": [
        "I was charged twice for the same transaction on my credit card.",
        "The transfer failed but my account was still debited and customer service was unresponsive."
    ]
})

def test_chunk_narratives():
    chunks, metadata = chunk_narratives(sample_df, text_column="Cleaned_Narrative")

    assert len(chunks) > 0
    assert len(chunks) == len(metadata)
    for meta in metadata:
        assert "complaint_id" in meta
        assert "product" in meta
        assert "text" in meta

def test_embedding_shape():
    chunks, metadata = chunk_narratives(sample_df, text_column="Cleaned_Narrative")
    texts = [meta["text"] for meta in metadata]

    vectors = embed_chunks(texts, model_name="all-MiniLM-L6-v2")

    assert isinstance(vectors, np.ndarray)
    assert vectors.shape[0] == len(texts)
    assert vectors.shape[1] == 384  # Embedding dimension of MiniLM-L6-v2

def test_faiss_indexing(tmp_path):
    chunks, metadata = chunk_narratives(sample_df, text_column="Cleaned_Narrative")
    texts = [meta["text"] for meta in metadata]
    vectors = embed_chunks(texts, model_name="all-MiniLM-L6-v2")

    index_path = tmp_path / "test_index.faiss"
    meta_path = tmp_path / "test_metadata.pkl"

    index_to_faiss(vectors, metadata, index_path=str(index_path), meta_path=str(meta_path))

    assert os.path.exists(index_path)
    assert os.path.exists(meta_path)

    index = faiss.read_index(str(index_path))
    assert index.ntotal == vectors.shape[0]

    with open(meta_path, "rb") as f:
        loaded_meta = pickle.load(f)
        assert isinstance(loaded_meta, list)
        assert loaded_meta[0]["text"] == metadata[0]["text"]
