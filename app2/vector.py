
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv("/Users/isurusenarathne/Documents/Dev/AINemo/app2/pizza_store_reviews.csv")

embeddings = OllamaEmbeddings(model="mxbai-embed-large")
db_location = os.path.join(os.getcwd(), "chroma_db")
if not os.path.exists(db_location):
    os.makedirs(db_location)

is_db_exists = os.path.exists(db_location)

if is_db_exists:
    docs = []
    ids = []

    for index, row in df.iterrows():
        doc = Document(
            page_content=row["review"] + " " + row["title"],
            metadata={
                "rating": row["rating"],
                "date": row["date"]
            }
        )
        docs.append(doc)
        ids.append(str(index))

    vector_store = Chroma(
        collection_name="pizza_reviews",
        embedding_function=embeddings,
        persist_directory=db_location
    )

    vector_store.add_documents(documents=docs, ids=ids)

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})   
