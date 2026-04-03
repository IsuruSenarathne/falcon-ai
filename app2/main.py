from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="qwen2.5-coder:7b")

template = """
you are expert in restaurant reviews.

Here are some reviews: {reviews}
Here is the question: {question}

Provide your answer according to following rules:
1. use response as {{ answer: "your answer", reasoning: "your reasoning" }}
2. your answer should have more details and be more comprehensive.
3. your reasoning should be detailed and explain how you arrived at the answer.
"""

promt = ChatPromptTemplate.from_template(template)
chain = promt | model

while True:
    print("\n---------------------------------")
    question = input("Enter your question: ")
    if (question.lower() == "exit"):
        break

    reviews = retriever.invoke(question)
    res = chain.invoke({
        "reviews": reviews,
        "question": question
    })
    print(res)
    print("\n---------------------------------\n")