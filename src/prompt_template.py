def build_prompt(context_chunks: list, question: str) -> str:
    """
    Constructs a structured prompt for financial complaint analysis using provided context chunks.
    
    The prompt is designed to guide a financial analyst assistant in answering questions based on
    customer complaint excerpts while maintaining strict context boundaries.

    Args:
        context_chunks (list): List of text excerpts containing customer complaint information
        question (str): The user's question to be answered based on the context

    Returns:
        str: A well-formatted prompt ready for submission to the language model

    Example:
        >>> chunks = ["Customer reported late fee issue...", "Another complaint about..."]
        >>> question = "What are common complaints about late fees?"
        >>> prompt = build_prompt(chunks, question)
    """
    # Format each context chunk with clear numbering and delimiters for readability
    context = "\n\n".join(
        [f"Excerpt {i+1}:\n\"\"\"\n{chunk.strip()}\n\"\"\"" 
         for i, chunk in enumerate(context_chunks)]
    )

    # Construct the complete prompt with:
    # 1. Role definition
    # 2. Task instructions
    # 3. Fallback behavior
    # 4. Formatted context
    # 5. The question
    prompt = (
        "You are a helpful and knowledgeable **financial analyst assistant** for CrediTrust.\n"
        "You will be shown several excerpts from real customer complaints.\n"
        "\n"
        "INSTRUCTIONS:\n"
        "1. Use only the provided excerpts to answer the user's question.\n"
        "2. Maintain a professional, analytical tone suitable for financial services.\n"
        "3. If different excerpts contain conflicting information, note this in your response.\n"
        "\n"
        "IMPORTANT LIMITATIONS:\n"
        "- If the answer cannot be found in the context, respond with:\n"
        "  \"I'm sorry, but the provided context doesn't contain enough information to answer that.\"\n"
        "- Never speculate or invent information beyond what's in the excerpts.\n"
        "\n"
        "CONTEXT EXCERPTS:\n"
        f"{context}\n\n"
        "QUESTION:\n"
        f"{question}\n\n"
        "REQUIRED RESPONSE FORMAT:\n"
        "Answer: [Your analysis here]"
    )
    
    return prompt