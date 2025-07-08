import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from src.retriever import ComplaintRetriever

@pytest.fixture
def mock_retriever():
    with patch("src.retriever.SentenceTransformer") as MockModel, \
         patch("src.retriever.faiss.read_index") as mock_faiss_read, \
         patch("builtins.open", create=True) as mock_open, \
         patch("pickle.load") as mock_pickle:

        # Mock sentence transformer output
        mock_model_instance = MockModel.return_value
        mock_model_instance.encode.return_value = np.random.rand(1, 384)  # Random 384-dim vector

        # Mock FAISS index search result
        mock_faiss_index = MagicMock()
        mock_faiss_index.search.return_value = (None, np.array([[0, 1, 2]]))
        mock_faiss_read.return_value = mock_faiss_index

        # Mock metadata
        mock_metadata = [
            {"text": "Complaint about credit card charges"},
            {"text": "Complaint about loan applications"},
            {"text": "Complaint about account closure"},
        ]
        mock_pickle.return_value = mock_metadata

        retriever = ComplaintRetriever("dummy.index", "dummy.pkl")
        yield retriever

def test_retrieve_top_k(mock_retriever):
    query = "Why are users unhappy with credit cards?"
    results = mock_retriever.retrieve(query, k=3)

    assert isinstance(results, list), "Expected results to be a list"
    assert len(results) == 3, f"Expected 3 results for k=3 but got {len(results)}"
    assert all("text" in r for r in results), "Each result should contain 'text' key"

def test_retrieve_default_k(mock_retriever):
    query = "Tell me about loans."
    results = mock_retriever.retrieve(query)

    # Default k=5, but mock only returns 3 metadata items
    assert len(results) <= 5
    assert results[0]["text"] == "Complaint about credit card charges"
