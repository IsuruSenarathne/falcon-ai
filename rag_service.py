"""
Core RAG Service Module
Handles all business logic for the RAG (Retrieval-Augmented Generation) system.
"""

import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


class RAGService:
    """Service class for RAG chain operations."""
    
    def __init__(self, data_file: str = "data.json"):
        """
        Initialize the RAG service.
        
        Args:
            data_file: Path to the JSON file containing the knowledge base.
        """
        load_dotenv()
        
        # Setup API Key
        os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
        
        # Load data from JSON file
        with open(data_file, "r") as f:
            data = json.load(f)["data"]
        
        # Create embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        
        # Create knowledge base
        knowledge_base = [f"Course: {title}. {desc}" for title, desc in data.items()]
        
        # Create vector store and retriever
        self.vectorstore = Chroma.from_texts(texts=knowledge_base, embedding=self.embeddings)
        self.retriever = self.vectorstore.as_retriever()
        
        # Define the RAG Prompt
        template = """Answer the question based ONLY on the following context:
{context}

Question: {question}
"""
        self.prompt = ChatPromptTemplate.from_template(template)
        
        # Initialize Model
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        
        # Create the RAG Chain
        self.rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def query(self, question: str) -> str:
        """
        Process a single question through the RAG chain.
        
        Args:
            question: The question to answer.
            
        Returns:
            The generated answer from the RAG chain.
            
        Raises:
            ValueError: If the question is empty.
            Exception: If there's an error processing the query.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        
        return self.rag_chain.invoke(question)
    
    def batch_query(self, questions: list) -> list:
        """
        Process multiple questions through the RAG chain.
        
        Args:
            questions: List of questions to answer.
            
        Returns:
            List of dictionaries containing questions and answers.
            
        Raises:
            ValueError: If questions list is empty or not a list.
        """
        if not isinstance(questions, list):
            raise ValueError("Questions must be a list")
        
        if not questions:
            raise ValueError("Questions list cannot be empty")
        
        results = []
        for question in questions:
            if not question or not question.strip():
                results.append({
                    "question": question,
                    "answer": None,
                    "error": "Question cannot be empty"
                })
            else:
                try:
                    answer = self.query(question)
                    results.append({
                        "question": question,
                        "answer": answer
                    })
                except Exception as e:
                    results.append({
                        "question": question,
                        "answer": None,
                        "error": str(e)
                    })
        
        return results
