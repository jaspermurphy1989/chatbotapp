from setuptools import setup, find_packages

setup(
    name="chatbot-app",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "langchain",
        "faiss-cpu",
        "chromadb",
        "streamlit",
        "openai",
        "sentence-transformers",
        "python-dotenv",
    ],
    python_requires=">=3.12",
)
