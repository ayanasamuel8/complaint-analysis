# Import necessary typing and data handling modules
from typing import List, Dict  # For type hints
import pandas as pd  # For data manipulation and DataFrame operations
from src.rag_pipeline import RAGPipeline  # Custom RAG pipeline implementation

class RAGEvaluator:
    """
    A class to evaluate the performance of a RAG (Retrieval-Augmented Generation) pipeline.
    It generates answers to questions, retrieves relevant sources, and creates evaluation reports.
    """
    
    def __init__(self, index_path: str, metadata_path: str):
        """
        Initialize the RAG evaluator with paths to the search index and metadata.
        
        Args:
            index_path (str): Path to the pre-built vector index for document retrieval
            metadata_path (str): Path to the metadata file containing document information
        """
        # Initialize the RAG pipeline with the provided index and metadata paths
        self.pipeline = RAGPipeline(index_path, metadata_path)

    def evaluate_questions(self, questions: List[str], k: int = 2) -> pd.DataFrame:
        """
        Evaluate the RAG pipeline's performance on a list of questions.
        
        Args:
            questions (List[str]): List of questions to evaluate the pipeline on
            k (int, optional): Number of top sources to display. Defaults to 2.
            
        Returns:
            pd.DataFrame: Results dataframe containing questions, answers, sources, and evaluation fields
        """
        results = []  # Store evaluation results for each question
        
        for q in questions:
            print(f"🔎 Evaluating: {q}")
            
            # Run the RAG pipeline to get answer and sources
            answer, sources = self.pipeline.run(q)
            
            # Display evaluation progress in console
            print("\nGenerated Answer:\n", answer)
            print("\nTop Retrieved Sources:\n", "\n---\n".join(sources[:k]))
            print("="*60)  # Separator for readability

            # Store results for this question
            results.append({
                "Question": q,
                "Generated Answer": answer,
                "Retrieved Sources": "\n---\n".join(sources[:k]),  # Format sources with separators
                "Quality Score (1-5)": "",  # Placeholder for manual quality assessment
                "Comments/Analysis": ""     # Placeholder for manual analysis notes
            })

        # Convert results to DataFrame for easier handling and reporting
        df = pd.DataFrame(results)
        return df

    def save_markdown(self, df: pd.DataFrame, path: str = "../report/rag_eval_report.md") -> None:
        """
        Save the evaluation results to a markdown file for documentation and sharing.
        
        Args:
            df (pd.DataFrame): DataFrame containing evaluation results
            path (str, optional): Output path for the markdown file. Defaults to "../report/rag_eval_report.md".
        """
        # Write DataFrame to markdown format with UTF-8 encoding
        with open(path, "w", encoding="utf-8") as f:
            f.write(df.to_markdown(index=False))  # Convert DataFrame to markdown table format
            
        print(f"📄 Markdown evaluation report saved to: {path}")