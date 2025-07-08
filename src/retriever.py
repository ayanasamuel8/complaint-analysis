import faiss  # Facebook's vector similarity search library
import pickle  # For serializing/deserializing Python objects
from sentence_transformers import SentenceTransformer  # For text embedding generation

class ComplaintRetriever:
    """
    A semantic search component for retrieving relevant financial complaints.
    
    Uses FAISS for efficient similarity search and SentenceTransformer for text embeddings.
    Designed to work with pre-processed complaint data and metadata.
    """

    def __init__(self, index_path: str, metadata_path: str, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the retriever with search index and embedding model.
        
        Args:
            index_path (str): Path to the FAISS index file
            metadata_path (str): Path to the pickled metadata file
            model_name (str): Name of the SentenceTransformer model to use. 
                            Defaults to "all-MiniLM-L6-v2" (good balance of speed/accuracy)
        
        Initializes:
            - Text embedding model
            - FAISS search index
            - Complaint metadata
        """
        # Initialize the sentence embedding model
        self.model = SentenceTransformer(model_name)
        
        # Load the FAISS index for efficient similarity search
        self.index = faiss.read_index(index_path)
        
        # Load the complaint metadata containing original text and additional information
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

    def retrieve(self, query: str, k: int = 5) -> list:
        """
        Retrieve the most relevant complaints for a given query.
        
        Args:
            query (str): The search query (e.g., customer question or complaint topic)
            k (int): Number of results to return. Defaults to 5.
            
        Returns:
            list: List of relevant complaint documents/metadata, ordered by relevance
            
        Process Flow:
        1. Encode query into embedding vector
        2. Search FAISS index for nearest neighbors
        3. Retrieve corresponding metadata for results
        """
        # Step 1: Convert query to embedding vector
        query_vec = self.model.encode([query])  # Returns numpy array
        
        # Step 2: Search FAISS index for similar vectors
        # Returns:
        # - D: Distances to nearest neighbors
        # - I: Indices of nearest neighbors
        _, I = self.index.search(query_vec, k)
        
        # Step 3: Map indices back to original complaint metadata
        return [self.metadata[i] for i in I[0]]  # I[0] because we only searched one query