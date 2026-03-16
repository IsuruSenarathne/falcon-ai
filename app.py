import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import json
from dotenv import load_dotenv

#load environment variables from .env file
load_dotenv()

# Load data from JSON file
with open("data.json", "r") as f:
    data = json.load(f)["data"]

# 1. Setup API Key
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# 2. Knowledge Base (The "Cheat Sheet")
# In a real app, this would be a PDF or Website

# 3. Create "Embeddings" (How AI reads/indexes text)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Instead of just passing the dictionary, we create a list of strings
# that includes BOTH the course name and the description.
knowledge_base = [f"Course: {title}. {desc}" for title, desc in data.items()]

# 4. Create a Vector Store (Our temporary database)
vectorstore = Chroma.from_texts(texts=knowledge_base, embedding=embeddings)
retriever = vectorstore.as_retriever()

# 5. Define the RAG Prompt
template = """Answer the question based ONLY on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 6. Initialize Model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# 7. The RAG Chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 8. Test it!
print("--- RAG RESPONSE ---")

print(rag_chain.invoke("who is the program advisor for DevOps and CI/CD Pipelines ?"))

# context_window_str = rag_chain.invoke("list of main keywords which help you to match the context? provide as comma separated string?");
# context_window = context_window_str.split(",")

