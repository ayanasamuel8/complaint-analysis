from src.retriever import ComplaintRetriever
from src.prompt_template import build_prompt
from src.generator import GeminiGenerator

class RAGPipeline:
    """
    A Retrieval-Augmented Generation (RAG) pipeline for financial complaint analysis.
    
    Combines document retrieval with generative AI to provide context-aware responses
    to customer service queries using financial complaint data.
    """

    def __init__(self, index_path: str, metadata_path: str):
        """
        Initialize the RAG pipeline components.
        
        Args:
            index_path (str): Path to the pre-built vector index for document retrieval
            metadata_path (str): Path to the metadata file containing document information
        """
        # Initialize the retriever for fetching relevant complaint documents
        self.retriever = ComplaintRetriever(index_path, metadata_path)
        
        # Initialize the generator (using Gemini model)
        self.generator = GeminiGenerator()  # ✅ Using Google's Gemini AI

    def run(self, query: str) -> tuple[str, list]:
        """
        Execute the complete RAG pipeline for a given query.
        
        Args:
            query (str): The user's question or search query
            
        Returns:
            tuple[str, list]: A tuple containing:
                - answer (str): The generated response
                - list: Top 2 context chunks used (for reporting/validation)
        
        Pipeline Steps:
        1. Retrieval: Fetch relevant document chunks
        2. Prompt Construction: Format the context and question
        3. Generation: Produce the final answer
        """
        # STAGE 1: DOCUMENT RETRIEVAL
        # Retrieve relevant complaint documents from the index
        docs = self.retriever.retrieve(query)
        
        # Extract text content from documents (handling both dict and string formats)
        chunks = [doc['text'] if isinstance(doc, dict) else doc for doc in docs]

        # STAGE 2: PROMPT ENGINEERING
        # Build a structured prompt incorporating the retrieved context
        prompt = build_prompt(chunks, query)

        # STAGE 3: RESPONSE GENERATION
        # Generate the final answer using Gemini
        answer = self.generator.generate(prompt)

        # Return both the answer and top 2 chunks for transparency and evaluation
        return answer, chunks[:2]