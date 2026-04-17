# 🚀 VedAI — AI-Powered Study Assistant

VedAI is an AI-powered study assistant designed to help students learn faster, smarter, and more interactively from their notes and PDFs.

It allows users to upload documents, ask questions, generate summaries, create mind maps, quizzes, and structured study plans using AI and vector search.

---

## 🌟 Features

- 🤖 AI Chat (context-aware)
- 📄 Upload PDF & ask questions
- 📝 Summarize text and documents
- 🧠 Generate mindmaps
- 🧪 Create quizzes (PDF + topic)
- 🎓 AI Tutor (step-by-step explanations)
- 📚 Study material & roadmap generator

---

## 🧠 Core Concept (RAG)

VedAI uses **Retrieval-Augmented Generation (RAG)**:

1. Upload PDF  
2. Extract text  
3. Split into chunks  
4. Convert into embeddings  
5. Store in vector DB (ChromaDB)  
6. Retrieve relevant chunks  
7. Send to AI → Generate answer  

---

## 🏗 Tech Stack

### 🔹 Frontend
- React (Vite)
- JavaScript
- ReactFlow (Mindmaps)
- CSS (Glassmorphism UI)

### 🔹 Backend
- FastAPI
- Python
- Groq API (LLaMA 3.1)
- ChromaDB (Vector Database)
- PyMuPDF (PDF Processing)

---

## 📁 Project Structure

```
vedai/
│
├── backend/
│   ├── main.py
│   ├── chroma_db/
│   └── .env
│
├── frontend/
│   ├── src/
│   ├── public/
│   └── package.json
│
└── README.md
```

---

## ⚙️ Backend Setup

### 1️⃣ Navigate to backend

```
cd backend
```

### 2️⃣ Create virtual environment

```
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 4️⃣ Add environment variables

Create `.env` file:

```
GROQ_API_KEY=your_api_key_here
```

### 5️⃣ Run backend

```
uvicorn main:app --reload
```

Backend runs at:
```
http://127.0.0.1:8000
```

---

## 🎨 Frontend Setup

### 1️⃣ Navigate to frontend

```
cd frontend
```

### 2️⃣ Install dependencies

```
npm install
```

### 3️⃣ Run project

```
npm run dev
```

Frontend runs at:
```
http://localhost:5173
```

---

## 🔗 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| /chat | POST | AI Chat |
| /upload | POST | Upload PDF |
| /ask | POST | Ask from PDF |
| /summarize | POST | Text summary |
| /mindmap | POST | Mindmap |
| /pdf-summary | GET | PDF summary |
| /pdf-mindmap | GET | PDF mindmap |
| /quiz | POST | Quiz from PDF |
| /quiz-topic | POST | Quiz from topic |
| /tutor | POST | AI tutor |
| /study-material | POST | Study plan |

---

## 🚀 Deployment

### 🔹 Backend (Render)

- Connect GitHub repo
- Add environment variable:

```
GROQ_API_KEY=your_api_key
```

- Start command:

```
uvicorn main:app --host 0.0.0.0 --port 10000
```

---

### 🔹 Frontend (Netlify)

#### Option 1: Drag & Drop

```
npm run build
```

Upload the `dist/` folder.

#### Option 2: GitHub

- Connect repo
- Build command:

```
npm run build
```

- Publish directory:

```
dist
```

---

## ⚠️ Limitations

- Requires internet connection
- Depends on AI accuracy
- Large PDFs may take time to process

---

## 🔮 Future Scope

- 🎤 Voice-based learning
- 📱 Mobile app
- 🌍 Multi-language support
- 🧠 Personalized learning tracking

---

## 🧠 Architecture

```
React Frontend
      ↓
FastAPI Backend
      ↓
Groq AI (LLaMA 3.1)
      ↓
ChromaDB (Vector Search)
```

---

## 👨‍💻 Author

Harshad Bandiwar

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!


## 👨‍💻 Live link

(https://vedai1.netlify.app/)

---
