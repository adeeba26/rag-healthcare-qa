from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

template = """You are a helpful medical information assistant.
Use ONLY the context below to answer the question.
If the answer is not in the context, say you do not have enough information.
Do not make up information.

Context:
{context}

Question: {question}

Answer:"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def ask(question):
    print("\nQ:", question)
    answer = chain.invoke(question)
    print("A:", answer)
    docs = retriever.invoke(question)
    print("\nSources:")
    for doc in docs:
        print(" -", doc.metadata.get("source"), "page", doc.metadata.get("page"))

if __name__ == "__main__":
    while True:
        q = input("\nAsk a question (or quit): ")
        if q.lower() == "quit":
            break
        ask(q)