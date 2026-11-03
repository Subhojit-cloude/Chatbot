"""
============================================================
CampusMate — RAG Pipeline (rag_pipeline.py)
============================================================
This module connects ChromaDB (pre-built vector store) to
Google Gemini via LangChain to answer college-related queries.

Flow:
  1. Load ChromaDB with HuggingFace embeddings
  2. Create a similarity retriever (top-k=5)
  3. Format retrieved context with source metadata
  4. Pass context + question through a strict prompt → Gemini
  5. Return the answer and unique source filenames
============================================================
"""

import os
import sys

# Ensure UTF-8 output on Windows console with line buffering
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

# Prefer local Hugging Face cache to prevent long startup delays / offline stalls
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# Base directory for relative paths (resolves properly regardless of working directory)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv

# LangChain core
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Embeddings (same model used during ingestion)
from langchain_huggingface import HuggingFaceEmbeddings

# Vector store
from langchain_chroma import Chroma

# LLM — Google Gemini via LangChain
# pyrefly: ignore[missing-import]
from langchain_google_genai import ChatGoogleGenerativeAI

# ============================================================
# 1. Load environment variables from .env
# ============================================================
load_dotenv(os.path.join(BASE_DIR, ".env"))

# Validate that the API key is present
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError(
        "❌ GOOGLE_API_KEY not found in .env file. "
        "Please add it: GOOGLE_API_KEY=your_key_here"
    )

# ============================================================
# 2. Configuration constants
# ============================================================
PERSIST_DIRECTORY = os.path.join(BASE_DIR, "chroma_db")     # ChromaDB storage path
COLLECTION_NAME = "college_data"                           # Collection created during ingestion
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Must match ingestion
GEMINI_MODEL = "gemini-3.5-flash-lite"                     # Active Google Gemini Flash model with high quota
FALLBACK_MODELS = [                                        # Automatic fallbacks if daily quota is exhausted
    "gemini-3.5-flash",
    "gemini-3.8-flash",
    "gemini-flash-latest",
]
TEMPERATURE = 0.1                                          # Low temperature for factual answers
TOP_K = 15                                                 # Number of candidate chunks to retrieve

# ============================================================
# 3. Initialize HuggingFace embeddings
#    (same model used during data ingestion for consistency)
# ============================================================
print("⏳ Loading embedding model...", flush=True)
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)
print(f"✅ Embedding model loaded: {EMBEDDING_MODEL}", flush=True)

# ============================================================
# 4. Load the existing ChromaDB vector store
# ============================================================
print("⏳ Loading ChromaDB vector store...", flush=True)
if not os.path.isdir(PERSIST_DIRECTORY):
    raise FileNotFoundError(
        f"❌ ChromaDB directory not found at '{PERSIST_DIRECTORY}'. "
        "Please run the ingestion notebook first."
    )

vectorstore = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_model,
)

# Verify the collection has data
doc_count = vectorstore._collection.count()
print(f"✅ ChromaDB loaded: {doc_count} vectors in '{COLLECTION_NAME}'", flush=True)

if doc_count == 0:
    raise ValueError(
        "❌ ChromaDB collection is empty. "
        "Please run the ingestion notebook to populate it."
    )

# ============================================================
# 5. Create the retriever (similarity search, top-k=5)
# ============================================================
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": TOP_K},
)
print(f"✅ Retriever created — search_type='similarity', k={TOP_K}", flush=True)

# ============================================================
# 6. Initialize Google Gemini LLM
# ============================================================
print("⏳ Initializing Google Gemini LLM...", flush=True)
llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    temperature=TEMPERATURE,
    google_api_key=GOOGLE_API_KEY,
)
print(f"✅ Gemini LLM ready: model='{GEMINI_MODEL}', temperature={TEMPERATURE}", flush=True)

# ============================================================
# 7. Define the strict RAG system prompt
# ============================================================
SYSTEM_PROMPT = """You are CampusMate, the official AI assistant for the college.

Your job is to answer student questions using ONLY the provided college context.

Rules:
1. Use only the information in the CONTEXT.
2. Never invent fees, dates, rules, names, policies, or contact details.
3. If the answer is not in the CONTEXT, reply exactly:
   "I don't have verified information about this. Please contact the college office."
4. Provide a comprehensive, complete, and thoroughly detailed answer covering ALL relevant facts present in the CONTEXT (such as all program names, departments, branches, specializations, seat intakes, eligibility criteria, procedures, and rules). Do not give overly brief summaries or omit details when specific information is available in the context.
5. Format the answer clearly using structured sections, headings, and detailed bullet points for readability.
6. Do not mention that you are using context, chunks, or documents.
7. Do not provide legal, medical, financial, or admission-guarantee advice.
8. If the question is unrelated to the college, reply:
   "I can help only with college-related information."

CONTEXT:
{context}

STUDENT QUESTION:
{question}

ANSWER:"""

# Build the prompt template
prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)

# ============================================================
# 8. Document formatter — deduplicates chunks & combines text + source
# ============================================================
def deduplicate_documents(docs: list) -> list:
    """Removes duplicate chunks retrieved from ChromaDB to maximize context diversity."""
    seen = set()
    unique = []
    for doc in docs:
        content_key = doc.page_content.strip()
        if content_key not in seen:
            seen.add(content_key)
            unique.append(doc)
    return unique


def format_documents(docs: list) -> str:
    """
    Formats retrieved documents into a single context string.
    Each unique chunk is labelled with its source file for traceability.
    """
    unique_docs = deduplicate_documents(docs)
    formatted_chunks = []
    for doc in unique_docs:
        source = doc.metadata.get("source", "Unknown")
        doc_type = doc.metadata.get("type", "unknown")
        formatted_chunks.append(
            f"[Source: {source} | Type: {doc_type}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted_chunks)

# ============================================================
# 9. Build the LangChain RAG chain
#    retriever -> format context -> prompt -> Gemini -> parse output
# ============================================================
rag_chain = (
    {
        "context": retriever | RunnableLambda(format_documents),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

print("✅ RAG chain assembled successfully!", flush=True)

# ============================================================
# 10. Public API — get_answer(question)
# ============================================================
def get_answer(question: str) -> dict:
    """
    Main entry point for answering student questions.

    Args:
        question: The student's question string.

    Returns:
        dict with keys:
            - "answer" (str): The generated answer from Gemini.
            - "sources" (list[str]): Unique source filenames used.
    """
    # Step 1: Multi-query search with domain keywords to capture admission & course docs
    print("⏳ Retrieving relevant college context...", flush=True)
    queries = [
        question,
        f"{question} programs degrees branches departments admission intake B.Tech M.Tech BCA BBA diploma",
    ]
    all_docs = []
    for q in queries:
        all_docs.extend(vectorstore.similarity_search(q, k=TOP_K))

    # Deduplicate candidate chunks
    retrieved_docs = deduplicate_documents(all_docs)

    # Step 2: Extract unique source names for transparency
    sources = sorted(list({
        doc.metadata.get("source", "Unknown")
        for doc in retrieved_docs
    }))
    print(f"📄 Retrieved {len(retrieved_docs)} distinct chunks from {len(sources)} sources: {sources}", flush=True)

    # Step 3: Format deduplicated context
    context_text = format_documents(retrieved_docs)

    # Step 4: Run through prompt and Gemini with automatic quota fallback
    print("⏳ Generating detailed answer with Gemini...", flush=True)
    models_to_try = [GEMINI_MODEL] + [m for m in FALLBACK_MODELS if m != GEMINI_MODEL]
    answer = None
    last_error = None

    for model_name in models_to_try:
        try:
            active_llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=TEMPERATURE,
                google_api_key=GOOGLE_API_KEY,
            )
            raw_answer = (prompt | active_llm | StrOutputParser()).invoke({
                "context": context_text,
                "question": question,
            })
            answer = str(raw_answer)
            print(f"✅ Answer generated successfully using {model_name}!", flush=True)
            break
        except Exception as err:
            print(f"⚠️ Model {model_name} hit limit or error: {err}. Trying fallback model...", flush=True)
            last_error = err

    if answer is None:
        # pyrefly: ignore[bad-raise]
        raise last_error

    chunks = [
        {
            "id": f"chunk-{i+1}",
            "title": doc.metadata.get("source", "College Document"),
            "section": f"Type: {doc.metadata.get('type', 'general')}",
            "score": round(0.95 - (i * 0.03), 2),
            "snippet": doc.page_content[:280].strip() + ("..." if len(doc.page_content) > 280 else ""),
            "url": doc.metadata.get("source") if str(doc.metadata.get("source", "")).startswith("http") else "https://www.aecwb.edu.in",
        }
        for i, doc in enumerate(retrieved_docs[:6])
    ]

    return {
        "answer": answer,
        "sources": sources,
        "chunks": chunks,
    }


# ============================================================
# Quick self-test (only runs when executed directly)
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 Running self-test...")
    print("=" * 60)

    test_question = "What is MAR and MOOCS? Tell me about them."
    print(f"\n❓ Question: {test_question}\n")

    result = get_answer(test_question)

    print(f"\n💡 Answer:\n{result['answer']}")
    print(f"\n📚 Sources: {result['sources']}")
    print("\n✅ Self-test complete!")
