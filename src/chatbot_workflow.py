# src/chatbot_workflow.py
from utils.nlp_utils import preprocess_text

def generate_response(user_input: str, session_id: str) -> str:
    """Process input and return bot response."""
    cleaned_input = preprocess_text(user_input)
    return f"Bot: You said '{cleaned_input}'"
