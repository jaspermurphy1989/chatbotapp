import streamlit as st
from src.document_loader import DocumentLoader
from src.vector_store import VectorStoreManager
from src.retriever import CustomRetriever
from src.chain import ChatbotChain
from src.utils import get_cache_key, load_from_cache, save_to_cache, clear_cache
from config import config
import os

# Set page config
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store_loaded" not in st.session_state:
    st.session_state.vector_store_loaded = False
if "chatbot_initialized" not in st.session_state:
    st.session_state.chatbot_initialized = False

# Sidebar for configuration
with st.sidebar:
    st.title("Configuration")
    
    # Vector store selection
    vector_store_type = st.radio(
        "Vector Store Type",
        ["FAISS", "Chroma"],
        index=0 if config.VECTOR_STORE_TYPE == "faiss" else 1,
    )
    config.VECTOR_STORE_TYPE = vector_store_type.lower()
    
    # Model selection
    llm_model = st.selectbox(
        "LLM Model",
        ["gpt-3.5-turbo", "gpt-4"],
        index=0 if config.LLM_MODEL == "gpt-3.5-turbo" else 1,
    )
    config.LLM_MODEL = llm_model
    
    # Data file selection
    data_file = st.selectbox(
        "Data File",
        [f.name for f in config.DATA_DIR.iterdir() if f.suffix == ".json"],
    )
    
    if st.button("Initialize/Reload Chatbot"):
        st.session_state.vector_store_loaded = False
        st.session_state.chatbot_initialized = False
        st.rerun()
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    if st.button("Clear Cache"):
        clear_cache()
        st.success("Cache cleared successfully!")
        st.rerun()

# Initialize the chatbot components
@st.cache_resource(show_spinner=False)
def initialize_chatbot(data_file_path: str):
    with st.spinner("Initializing chatbot components..."):
        # Step 1: Load and process documents
        loader = DocumentLoader()
        raw_docs = loader.load_json_documents(data_file_path)
        processed_docs = loader.process_documents(raw_docs)
        split_docs = loader.split_documents(processed_docs)
        
        # Step 2: Create or load vector store
        vector_store_manager = VectorStoreManager()
        vector_store = vector_store_manager.load_vector_store()
        
        if vector_store is None:
            vector_store = vector_store_manager.create_vector_store(split_docs)
            vector_store_manager.save_vector_store(vector_store)
        
        # Step 3: Create retriever and chain
        retriever = CustomRetriever(vector_store)
        chatbot = ChatbotChain(retriever)
        
        st.session_state.vector_store_loaded = True
        st.session_state.chatbot_initialized = True
        
        return chatbot

# Main chat interface
st.title("RAG Chatbot")
st.write("Ask me anything based on the provided documents!")

# Only initialize if not already done
if not st.session_state.chatbot_initialized:
    data_file_path = str(config.DATA_DIR / data_file)
    chatbot = initialize_chatbot(data_file_path)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Check cache first
    cache_key = get_cache_key(prompt)
    cached_response = load_from_cache(cache_key)
    
    if cached_response:
        response = cached_response["response"]
    else:
        # Get chatbot response
        with st.spinner("Thinking..."):
            result = chatbot.invoke(prompt)
            response = result["response"]
            save_to_cache(cache_key, {"response": response})
    
    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
