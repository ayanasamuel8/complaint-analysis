import os
from dotenv import load_dotenv  # For loading environment variables from .env file
import google.generativeai as genai  # Google's Gemini AI SDK

class GeminiGenerator:
    """
    A class to interact with Google's Gemini AI model for text generation.
    Handles API configuration, chat session management, and response generation.
    """

    def __init__(self, model_name: str = "models/gemini-2.5-pro") -> None:
        """
        Initialize the Gemini generator with API configuration.
        
        Args:
            model_name (str): Name of the Gemini model to use. 
                            Defaults to "models/gemini-2.5-pro".
        
        Raises:
            ValueError: If the API key is not found in environment variables.
        """
        # Load environment variables from .env file
        load_dotenv()
        
        # Retrieve API key from environment variables
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file")
        
        # Configure the Gemini API with the obtained key
        genai.configure(api_key=api_key)
        
        # Initialize the generative model and start a chat session
        self.model = genai.GenerativeModel(model_name)
        self.chat = self.model.start_chat()  # Maintains conversation history

    def generate(self, prompt: str) -> str:
        """
        Generate a response from Gemini based on the given prompt.
        
        Args:
            prompt (str): The input text/prompt to send to Gemini.
            
        Returns:
            str: The generated response text, or error message if generation fails.
        """
        try:
            # Send the prompt to Gemini and get the response
            response = self.chat.send_message(prompt)
            
            # Return cleaned response text
            return response.text.strip()
            
        except Exception as e:
            # Log the error and re-raise the exception
            logging.error("❌ Error generating with Gemini: %s", str(e))
            raise