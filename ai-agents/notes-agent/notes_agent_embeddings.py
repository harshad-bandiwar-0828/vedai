import fitz  # PyMuPDF
import chromadb
from chromadb.utils import embedding_functions
import re


# ==============================
# 1. Extract text
# ==============================
def extract_text_from_pdf(file_path):
    text = ""
    pdf = fitz.open(file_path)

    for page in pdf:
        text += page.get_text()

    return text


# ==============================
# 2. Clean text
# ==============================
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text


# ==============================
# 3. Chunk text
# ==============================
def chunk_text(text, chunk_size=80):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


# ==============================
# 4. Setup DB (FIXED PATH)
# ==============================
client = chromadb.PersistentClient(
    path="D:/vedai/chroma_db"   # ✅ SAME PATH EVERYWHERE
)

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.get_or_create_collection(
    name="vedai_notes",
    embedding_function=embedding_function
)

# ==============================
# 5. Process PDF
# ==============================
def process_pdf(file_path):
    print(f"\n📄 Processing: {file_path}")

    raw_text = extract_text_from_pdf(file_path)

    if not raw_text.strip():
        print("❌ No text found")
        return

    text = clean_text(raw_text)
    chunks = chunk_text(text)

    print(f"✂️ Created {len(chunks)} chunks")

    for i, chunk in enumerate(chunks):
        collection.add(
            ids=[f"{file_path}_{i}"],
            documents=[chunk],
            metadatas=[{"source": file_path}]
        )

    print("✅ Stored in DB!")


# ==============================
# 6. Run
# ==============================
if __name__ == "__main__":
    file_path = input("Enter PDF path: ")
    process_pdf(file_path)