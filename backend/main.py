from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import chromadb
from chromadb.utils import embedding_functions
import fitz
import os

from groq import Groq
from dotenv import load_dotenv

# ===============================
# 🔐 LOAD ENV
# ===============================
load_dotenv()
client_ai = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ===============================
# 📁 ACTIVE FILE
# ===============================
active_file = None

# ===============================
# 🚀 APP
# ===============================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# 📦 MODELS
# ===============================
class QueryRequest(BaseModel):
    query: str

class TextRequest(BaseModel):
    text: str

class QuizRequest(BaseModel):
    num_questions: int = 5

class TopicRequest(BaseModel):
    topic: str
    level: str = "beginner"
    
class FeedbackRequest(BaseModel):
    questions: list
    answers: list

# ✅ NEW (CHAT MEMORY)
class ChatRequest(BaseModel):
    messages: list

# ===============================
# 🧠 DB (FIXED - LAZY LOAD)
# ===============================
client = chromadb.PersistentClient(path="./chroma_db")

collection = None

def get_collection():
    global collection

    if collection is None:
        from chromadb.utils import embedding_functions

        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        collection = client.get_or_create_collection(
            name="vedai_notes",
            embedding_function=embedding_function
        )

    return collection

# ===============================
# 🏠 HOME
# ===============================
@app.get("/")
def home():
    return {"message": "VedAI Backend Running 🚀"}

# ===============================
# 🤖 CHAT (WITH MEMORY)
# ===============================
@app.post("/chat")
def chat(req: ChatRequest):
    try:
        res = client_ai.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=req.messages
        )
        return {"answer": res.choices[0].message.content.strip()}
    except Exception as e:
        return {"answer": f"❌ {str(e)}"}

# ===============================
# 📄 ASK (PDF QA)
# ===============================
@app.post("/ask")
def ask(req: QueryRequest):
    global active_file

    if not active_file:
        return {"answer": "❌ No PDF selected"}

    col = get_collection()
    results = col.query(
        query_texts=[req.query],
        n_results=3,
        where={"file": active_file}
    )

    docs = results.get("documents", [])

    if not docs or not docs[0]:
        return {"answer": "❌ No relevant data found"}

    context = "\n".join(docs[0])

    prompt = f"""
Answer ONLY from this context.

Context:
{context}

Question:
{req.query}

If not found, say: Not found in notes.
Keep answer short.
"""

    res = client_ai.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return {"answer": res.choices[0].message.content.strip()}

# ===============================
# 📝 TEXT SUMMARY
# ===============================
@app.post("/summarize")
def summarize(req: TextRequest):
    res = client_ai.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{
            "role": "user",
            "content": f"Summarize into 5 bullet points:\n{req.text}"
        }]
    )
    return {"summary": res.choices[0].message.content.strip()}

# ===============================
# 🧩 TEXT MINDMAP
# ===============================
@app.post("/mindmap")
def mindmap(req: TextRequest):
    res = client_ai.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{
            "role": "user",
            "content": f"""
Convert into mindmap format:

Main Topic
- Subtopic
  - Detail

Text:
{req.text}
"""
        }]
    )
    return {"mindmap": res.choices[0].message.content.strip()}

# ===============================
# 📄 PDF SUMMARY
# ===============================
@app.get("/pdf-summary")
def pdf_summary():
    global active_file

    if not active_file:
        return {"summary": "❌ No PDF selected"}

    col = get_collection()
    results = col.query(
        query_texts=["summary"],
        n_results=10,
        where={"file": active_file}
    )

    docs = results.get("documents", [])

    if not docs or not docs[0]:
        return {"summary": "❌ No data found"}

    text = " ".join(docs[0])

    res = client_ai.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{
            "role": "user",
            "content": f"Summarize into 5 bullet points:\n{text}"
        }]
    )

    return {"summary": res.choices[0].message.content.strip()}

# ===============================
# 🧩 PDF MINDMAP
# ===============================
@app.get("/pdf-mindmap")
def pdf_mindmap():
    global active_file

    if not active_file:
        return {"mindmap": "❌ No PDF selected"}
    
    col = get_collection()
    results = col.query(
        query_texts=["main topics"],
        n_results=10,
        where={"file": active_file}
    )

    docs = results.get("documents", [])

    if not docs or not docs[0]:
        return {"mindmap": "❌ No data found"}

    text = " ".join(docs[0])

    res = client_ai.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{
            "role": "user",
            "content": f"""
Convert into mindmap:

Main Topic
- Subtopic
  - Detail

{text}
"""
        }]
    )

    return {"mindmap": res.choices[0].message.content.strip()}

# ===============================
# 🧪 QUIZ FROM PDF
# ===============================
@app.post("/quiz")
def quiz(req: QuizRequest):
    global active_file

    if not active_file:
        return {"quiz": "❌ No PDF selected"}

    col = get_collection()
    results = col.query(
        query_texts=["important concepts"],
        n_results=15,
        where={"file": active_file}
    )

    docs = results.get("documents", [])

    if not docs or not docs[0]:
        return {"quiz": "❌ No data found"}

    context = " ".join(docs[0])

    prompt = f"""
You are an AI exam generator.

STRICT RULES:
- Generate questions ONLY from the given context
- DO NOT use outside knowledge
- Questions must be directly based on the content
- If context is insufficient, create fewer questions

======================
CONTEXT:
{context}
======================

Create {req.num_questions} MCQs.

FORMAT:

Q1: Question
A. option
B. option
C. option
D. option
Answer: A
Explanation: Based on context

Q2: ...

IMPORTANT:
- Questions must match context exactly
- No generic questions
- No hallucination

"""

    res = client_ai.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return {"quiz": res.choices[0].message.content.strip()}

# ===============================
# 🧠 QUIZ FROM TOPIC
# ===============================
@app.post("/quiz-topic")
def quiz_topic(req: TopicRequest):
    prompt = f"""

Create a professional multiple choice quiz on: {req.topic}

Rules:
- Generate exactly {10} questions
- Each question must have 4 options (A, B, C, D)
- DO NOT show the correct answer inside the question
- Provide answers separately in a JSON format

Format STRICTLY like this:

QUESTIONS:
Q1. Question text
A. Option
B. Option
C. Option
D. Option

Q2. Question text
A. Option
B. Option
C. Option
D. Option

---

ANSWERS (JSON ONLY):
[
  {{"question": 1, "correct": "B", "explanation": "Explain why"}},
  {{"question": 2, "correct": "A", "explanation": "Explain why"}}
]

Important:
- Do NOT mix answers with questions
- Do NOT write "Answer:" in questions section
- Output must be clean and structured

"""

    res = client_ai.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return {"quiz": res.choices[0].message.content.strip()}

# ===============================
# 🧠 QUIZ FEEDBACK
# ===============================
@app.post("/quiz-feedback")
def quiz_feedback(req: FeedbackRequest):
    prompt = f"""
Analyze performance:

Questions:
{req.questions}

Answers:
{req.answers}

Give:
- Score
- Weak areas
- Tips
"""

    res = client_ai.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return {"feedback": res.choices[0].message.content.strip()}

# ===============================
# 📤 UPLOAD PDF
# ===============================
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    global active_file

    try:
        active_file = file.filename
        path = f"temp_{file.filename}"

        with open(path, "wb") as f:
            f.write(await file.read())

        pdf = fitz.open(path)
        text = ""

        for p in pdf:
            text += p.get_text()

        pdf.close()

        words = text.split()

        chunks = [
            " ".join(words[i:i + 80])
            for i in range(0, len(words), 80)
        ]

        for i, chunk in enumerate(chunks):
            col = get_collection()
            col.add(
                ids=[f"{file.filename}_{i}"],
                documents=[chunk],
                metadatas=[{"file": file.filename}]
            )

        os.remove(path)

        return {"message": f"✅ {file.filename} uploaded"}

    except Exception as e:
        return {"error": str(e)}

# ===============================
# 🧠 AI TUTOR
# ===============================
class TutorRequest(BaseModel):
    messages: list

@app.post("/tutor")
def tutor(req: TutorRequest):
    try:
        res = client_ai.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": """
You are VedAI Tutor.

Teach like a friendly teacher.

Rules:
- Explain in SIMPLE language
- Give real-life examples
- Break into steps
- Ask 1 follow-up question
- Encourage student

Do NOT give boring answers.
"""
                }
            ] + req.messages
        )

        return {"answer": res.choices[0].message.content.strip()}

    except Exception as e:
        return {"answer": f"❌ {str(e)}"}

# ===============================
# 📚 STUDY MATERIAL (ADVANCED)
# ===============================
@app.post("/study-material")
def study_material(req: TopicRequest):
    try:
        prompt = f"""
Suggest BEST study resources for: {req.topic}

RULES:
- Focus on high-quality, trusted, and relevant resources
- Prefer globally trusted + India-friendly platforms
- Keep it useful for students (beginner → advanced)

========================

📘 Important Topics:
- List 5–7 key concepts

📄 Notes & Articles:
- Give 5–7 BEST direct links from trusted sources

🌐 Websites:
- Give 5–7 platforms to learn properly

🎥 YouTube:
- Give ONLY search links (not random videos)
- Include:
  • lectures
  • full course
  • playlists
- Format: 
  - https://www.youtube.com/results?search_query={req.topic}+lecture
  - https://www.youtube.com/results?search_query={req.topic}+full+course
  - https://www.youtube.com/results?search_query={req.topic}+playlist

📚 Books:
- 2–3 best books (name + author)

🎯 Roadmap:
10 points 
1. Start with basics
2. Learn core topics
3. Practice problems
4. Revise & test

========================

IMPORTANT:
- ONLY real links
- NO explanation
- Clean formatting
"""

        res = client_ai.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )

        return {"result": res.choices[0].message.content.strip()}

    except Exception as e:
        return {"result": f"❌ {str(e)}"}

@app.post("/study-material")
def study_material(req: TopicRequest):
    level = req.level

    prompt = f"""
Create a structured study plan for: {req.topic}

Level: {level}

========================

📘 Concepts:
- Based on level

📄 Notes:
- Give real links

🌐 Websites:
- Trusted platforms

🎥 YouTube:
- https://www.youtube.com/results?search_query={req.topic}+lecture
- https://www.youtube.com/results?search_query={req.topic}+full+course
- https://www.youtube.com/results?search_query={req.topic}+playlist

🧪 Practice:
- 3 practice questions

🎯 Roadmap:
1. Basics
2. Core concepts
3. Practice
4. Revision

⏱ Time Required:
- Estimate learning time

========================

IMPORTANT:
- Clean format
- Only real links
- No explanation
"""

    res = client_ai.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return {"study": res.choices[0].message.content.strip()}