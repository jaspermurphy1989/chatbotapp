from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.retriever import CustomRetriever
from src.constants import Constants
from config import config
from typing import Dict

class ChatbotChain:
    def __init__(self, retriever: CustomRetriever):
        self.retriever = retriever
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=0.7,
            api_key=config.OPENAI_API_KEY,
        )
        self.output_parser = StrOutputParser()
        
        # Define the prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", Constants.SYSTEM_PROMPT),
            ("human", """
            Context: {context}
            
            Question: {question}
            
            Please provide a detailed answer based on the context provided.
            If the context doesn't contain the answer, say "I don't have that information.".
            """),
        ])

    def format_docs(self, docs):
        """Format documents for the prompt."""
        return "\n\n".join(doc.page_content for doc in docs)

    def create_chain(self):
        """Create the RAG chain."""
        return (
            {
                "context": self.retriever | self.format_docs,
                "question": RunnablePassthrough(),
            }
            | self.prompt
            | self.llm
            | self.output_parser
        )

    def invoke(self, query: str) -> Dict:
        """Invoke the chain with a query."""
        chain = self.create_chain()
        return {"query": query, "response": chain.invoke(query)}
