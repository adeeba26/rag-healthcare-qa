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

test_cases = [
    {
        "question": "What are the symptoms of Alzheimer's disease?",
        "expected_keywords": ["memory", "confusion", "behavior"]
    },
    {
        "question": "How many people are affected by dementia worldwide?",
        "expected_keywords": ["million", "people", "worldwide"]
    },
    {
        "question": "What is the difference between Alzheimer's and dementia?",
        "expected_keywords": ["dementia", "Alzheimer"]
    },
    {
        "question": "What treatments exist for Alzheimer's disease?",
        "expected_keywords": ["treatment", "medication", "care"]
    },
    {
        "question": "What causes Alzheimer's disease?",
        "expected_keywords": ["brain", "plaques", "neurons"]
    },
]

print("=" * 60)
print("RAG PIPELINE EVALUATION")
print("=" * 60)

results = []

for i, case in enumerate(test_cases, 1):
    question = case["question"]
    keywords = case["expected_keywords"]

    answer = chain.invoke(question)
    docs = retriever.invoke(question)

    keyword_hits = sum(
        1 for kw in keywords
        if kw.lower() in answer.lower()
    )
    keyword_score = keyword_hits / len(keywords)

    has_source = len(docs) > 0
    not_hallucinated = "do not have enough information" not in answer.lower()

    print(f"\nTest {i}: {question}")
    print(f"Answer: {answer[:200]}...")
    print(f"Keyword match:     {keyword_score:.0%} ({keyword_hits}/{len(keywords)} keywords found)")
    print(f"Sources retrieved: {'YES' if has_source else 'NO'} ({len(docs)} chunks)")
    print(f"Has answer:        {'YES' if not_hallucinated else 'NO ANSWER FOUND'}")

    results.append({
        "keyword_score": keyword_score,
        "has_source": has_source,
        "not_hallucinated": not_hallucinated
    })

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
avg_keyword = sum(r["keyword_score"] for r in results) / len(results)
source_rate = sum(r["has_source"] for r in results) / len(results)
answer_rate = sum(r["not_hallucinated"] for r in results) / len(results)

print(f"Avg keyword match score: {avg_keyword:.0%}")
print(f"Source retrieval rate:   {source_rate:.0%}")
print(f"Answer rate:             {answer_rate:.0%}")
print("=" * 60)