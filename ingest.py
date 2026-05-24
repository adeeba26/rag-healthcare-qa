import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

print("Loading documents...")
loader = PyPDFDirectoryLoader("docs/")
documents = loader.load()
print(f"  Loaded {len(documents)} pages")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)
chunks = splitter.split_documents(documents)
print(f"  Split into {len(chunks)} chunks")

print("Embedding chunks (first time downloads ~90MB, be patient)...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(chunks, embeddings)

os.makedirs("vectorstore", exist_ok=True)
vectorstore.save_local("vectorstore")
print("Done! Vector store saved.")
