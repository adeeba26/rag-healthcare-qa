import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

st.set_page_config(
    page_title="Healthcare RAG Assistant",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Healthcare Q&A Assistant")
st.caption("Ask questions about Alzheimer's disease and dementia. Answers are grounded in medical documents.")

@st.cache_resource
def load_chain():
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
    return chain, retriever

chain, retriever = load_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Ask a medical question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            answer = chain.invoke(question)
            docs = retriever.invoke(question)

        st.markdown(answer)

        with st.expander("Sources used"):
            for doc in docs:
                src = doc.metadata.get("source", "unknown")
                page = doc.metadata.get("page", "?")
                st.caption(f"📄 {src} — page {page}")
                st.text(doc.page_content[:200] + "...")

    st.session_state.messages.append({"role": "assistant", "content": answer})