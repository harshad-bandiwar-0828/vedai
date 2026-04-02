import chromadb
from chromadb.utils import embedding_functions
from groq import Groq


# ==============================
# 1. API KEY
# ==============================
import os
groq_api_key = os.getenv("GROQ_API_KEY")

# ==============================
# 2. Setup DB
# ==============================
client = chromadb.PersistentClient(path="./chroma_db")

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.get_or_create_collection(
    name="vedai_notes",
    embedding_function=embedding_function
)

print("📊 Documents in DB:", collection.count())


# ==============================
# 3. Model fallback system
# ==============================
MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile",
    "gemma-7b-it"
]


# ==============================
# 4. QA function
# ==============================
def answer_question(query):

    results = collection.query(
        query_texts=[query],
        n_results=3
    )

    docs = results["documents"][0]

    if not docs:
        return "❌ No relevant notes found."

    context = "\n".join(docs)

    prompt = f"""
Answer the question based ONLY on the context below.

Context:
{context}

Question:
{query}

Answer in 2-3 lines:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
       messages=[
    {"role": "system", "content": "You are VedAI. Answer ONLY from notes."},
    {"role": "user", "content": prompt}
]
    )

    return response.choices[0].message.content.strip()


# ==============================
# 5. Run
# ==============================
if __name__ == "__main__":
    query = input("Ask your question: ")
    answer = answer_question(query)

    print("\n🤖 VedAI Answer:\n")
    print(answer)