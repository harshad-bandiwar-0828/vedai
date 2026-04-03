import { useState, useRef, useEffect } from "react";
import "./App.css";
import MindmapGraph from "./components/MindmapGraph";
import Quiz from "./components/Quiz";
import StudyMaterial from "./components/StudyMaterial";
import HomePage from "./pages/HomePage";
import "./pages/HomePage.css";

function App() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState("home");
  const [isTyping, setIsTyping] = useState(false);
  const [chatHistory, setChatHistory] = useState([{ role: "system", content: "You are VedAI, a helpful AI tutor." }]);
  const [notesHistory, setNotesHistory] = useState([]);
  const [tutorHistory, setTutorHistory] = useState([{ role: "system", content: "You are VedAI Tutor." }
]);
  const [level, setLevel] = useState("beginner");
  const chatEndRef = useRef(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);

useEffect(() => {
  const move = (e) => {
    document.body.style.setProperty("--x", e.clientX + "px");
    document.body.style.setProperty("--y", e.clientY + "px");
  };

  window.addEventListener("mousemove", move);
  return () => window.removeEventListener("mousemove", move);
}, []);

  // ✅ Smooth scroll
  useEffect(() => {
    if (!isTyping) {
      chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, isTyping]);

  // ===============================
  // ADD MESSAGE
  // ===============================
  const addMessage = (sender, text, type = "text") => {
    setMessages((prev) => [...prev, { sender, text, type }]);
  };

  // ===============================
  // ✨ TYPING (ONLY FOR TEXT)
  // ===============================
  const typeMessage = (text, type = "text") => {
    // ❌ No typing animation for structured data
    if (type === "mindmap" || type === "quiz" || type === "study")  {
      setMessages((prev) => [...prev, { sender: "ai", text, type }]);
      return;
    }

    let index = 0;
    setIsTyping(true);

    setMessages((prev) => [...prev, { sender: "ai", text: "", type }]);

    const speed = text.length > 300 ? 3 : 8;

    const interval = setInterval(() => {
      index++;

      setMessages((prev) => {
        const lastIndex = prev.length - 1;

        return prev.map((msg, i) =>
          i === lastIndex
            ? { ...msg, text: text.slice(0, index) }
            : msg
        );
      });

      if (index >= text.length) {
        clearInterval(interval);
        setIsTyping(false);
      }
    }, speed);
  };

  // ================= CHAT =================
  const askChat = async () => {
  if (!input.trim()) return;

  const userMessage = {
    role: "user",
    content: input
  };

  addMessage("user", input);

  // 🧠 Add user message to history
  const updatedHistory = [...chatHistory, userMessage];
  setChatHistory(updatedHistory);

  setLoading(true);

  try {
    const res = await fetch("https://vedai-backend-78tx.onrender.com/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        messages: updatedHistory  // ✅ FULL HISTORY
      })
    });

    const data = await res.json();

    // 🧠 Add AI response to history
    const aiMessage = {
      role: "assistant",
      content: data.answer
    };

    setChatHistory([...updatedHistory, aiMessage]);

    typeMessage(data.answer);

  } catch (err) {
    console.error(err);
    addMessage("ai", "❌ Chat error");
  }

  setInput("");
  setLoading(false);
};

  // ================= NOTES =================
  const askNotes = async () => {
  if (!input.trim()) return;

  addMessage("user", input);

  // 🧠 Add user question to history
  const updatedHistory = [...notesHistory, `User: ${input}`];
  setNotesHistory(updatedHistory);

  setLoading(true);

  try {
    const res = await fetch("https://vedai-backend-78tx.onrender.com/ask", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        query: updatedHistory.join("\n")   // ✅ FULL CONTEXT
      })
    });

    const data = await res.json();

    // 🧠 Save AI response
    setNotesHistory([...updatedHistory, `AI: ${data.answer}`]);

    typeMessage(data.answer);

  } catch {
    addMessage("ai", "❌ Notes error");
  }

  setInput("");
  setLoading(false);
};

  // ================= SUMMARY =================
  const summarize = async () => {
    if (!input.trim()) return;

    addMessage("user", input);
    setLoading(true);

    try {
      const res = await fetch("https://vedai-backend-78tx.onrender.com/summarize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: input })
      });

      const data = await res.json();
      typeMessage(data.summary);
    } catch {
      addMessage("ai", "❌ Summary error");
    }

    setInput("");
    setLoading(false);
  };

  // ================= MINDMAP =================
  const mindmap = async () => {
    if (!input.trim()) return;

    addMessage("user", input);
    setLoading(true);

    try {
      const res = await fetch("https://vedai-backend-78tx.onrender.com/mindmap", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: input })
      });

      const data = await res.json();
      typeMessage(data.mindmap, "mindmap");
    } catch {
      addMessage("ai", "❌ Mindmap error");
    }

    setInput("");
    setLoading(false);
  };

  // ================= PDF SUMMARY =================
  const summarizePDF = async () => {
    setLoading(true);

    try {
      const res = await fetch("https://vedai-backend-78tx.onrender.com/pdf-summary");
      const data = await res.json();
      typeMessage(data.summary);
    } catch {
      addMessage("ai", "❌ PDF Summary error");
    }

    setLoading(false);
  };

  // ================= PDF MINDMAP =================
  const mindmapPDF = async () => {
    setLoading(true);

    try {
      const res = await fetch("https://vedai-backend-78tx.onrender.com/pdf-mindmap");
      const data = await res.json();
      typeMessage(data.mindmap, "mindmap");
    } catch {
      addMessage("ai", "❌ PDF Mindmap error");
    }

    setLoading(false);
  };

  // ================= QUIZ PDF =================
  const generateQuizPDF = async () => {
    setLoading(true);

    try {
      const res = await fetch("https://vedai-backend-78tx.onrender.com/quiz", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ num_questions: 5 })
      });

      const data = await res.json();
      typeMessage(data.quiz, "quiz");
    } catch {
      addMessage("ai", "❌ Quiz error");
    }

    setLoading(false);
  };

  // ================= QUIZ TOPIC =================
  const generateQuizTopic = async () => {
    if (!input.trim()) return;

    addMessage("user", input);
    setLoading(true);

    try {
      const res = await fetch("https://vedai-backend-78tx.onrender.com/quiz-topic", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ topic: input })
      });

      const data = await res.json();
      typeMessage(data.quiz, "quiz");
    } catch {
      addMessage("ai", "❌ Topic quiz error");
    }

    setInput("");
    setLoading(false);
  };

  // ================= UPLOAD =================
  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);

    try {
      const res = await fetch("https://vedai-backend-78tx.onrender.com/upload", {
        method: "POST",
        body: formData
      });

      const data = await res.json();
      addMessage("ai", data.message);
    } catch {
      addMessage("ai", "❌ Upload error");
    }

    setLoading(false);
  };

  const askTutor = async () => {
  if (!input.trim()) return;

  const userMsg = { role: "user", content: input };

  addMessage("user", input);

  const updated = [...tutorHistory, userMsg];
  setTutorHistory(updated);

  setLoading(true);

  try {
    const res = await fetch("https://vedai-backend-78tx.onrender.com/tutor", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        messages: updated
      })
    });

    const data = await res.json();

    const aiMsg = { role: "assistant", content: data.answer };
    setTutorHistory([...updated, aiMsg]);

    typeMessage(data.answer);

  } catch {
    addMessage("ai", "❌ Tutor error");
  }

  setInput("");
  setLoading(false);
};

const getStudyMaterial = async () => {
  if (!input.trim()) return;

  addMessage("user", input);
  setLoading(true);

  try {
    const res = await fetch("https://vedai-backend-78tx.onrender.com/study-material", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ 
        topic: input,
        level: level   // ✅ FIXED
      })
    });

    const data = await res.json();

    typeMessage(data.result, "study"); // ✅ FIXED

  } catch {
    addMessage("ai", "❌ Study material error");
  }

  setInput("");
  setLoading(false);
};
 
 return (
  <div className="app">

    {/* ✅ HOMEPAGE ONLY */}
    {activeTab === "home" && (
      <HomePage setActiveTab={setActiveTab} />
    )}

    {/* ✅ HIDE EVERYTHING WHEN HOME */}
    {activeTab !== "home" && (
      <>
        {/* SIDEBAR */}
        <div className={`sidebar ${sidebarOpen ? "open" : ""}`}>
          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            <img src="/logo.png" alt="VedAI" width="30" />
            <h2>VedAI</h2>
          </div>

          <button onClick={() => setActiveTab("home")}>🏠 Home</button>

          <button className={activeTab === "chat" ? "active" : ""}
          onClick={() => {
            setActiveTab("chat");
            setSidebarOpen(false);
            setMessages([]);
            setChatHistory([
              { role: "system", content: "You are VedAI, a helpful AI tutor." }
            ]);
          }}>
            🤖 Chat
          </button>


          <button className={activeTab === "notes" ? "active" : ""}
           onClick={() => {
            setActiveTab("notes");
            setMessages([]);
            setNotesHistory([]);
            setSidebarOpen(false);
          }}>
            📄 Notes
          </button>

          <button className={activeTab === "quiz" ? "active" : ""}
          onClick={() => {
            setActiveTab("quiz");
            setMessages([]);
            setSidebarOpen(false);
          }}>
            🧪 Quiz
          </button>

          <button className={activeTab === "tutor" ? "active" : ""} 
          onClick={() => {
            setActiveTab("tutor");
            setMessages([]);
            setTutorHistory([
              { role: "system", content: "You are VedAI Tutor." }
            ]);
            setSidebarOpen(false);
          }}>
            🧠 Tutor
          </button>

          <button className={activeTab === "study" ? "active" : ""} 
          onClick={() => {
            setActiveTab("study");
            setMessages([]);
            setSidebarOpen(false);
          }}>
            📚 Study
          </button>
        </div>

        {/* MAIN */}
        <div className="main">

          <button 
  className="menu-btn"
  onClick={() => setSidebarOpen(!sidebarOpen)}
>
  ☰
</button>

          {/* CHAT BOX */}
          <div className="chat-box">
           {messages.map((msg, index) => {

  // 🧠 MINDMAP
  if (msg.type === "mindmap") {
    return (
      <div key={index} className={`message ${msg.sender}`}>
        {isTyping ? (
          <div>🧠 Generating mindmap...</div>
        ) : (
          <div style={{ width: "100%", height: "400px" }}>
            <MindmapGraph text={msg.text || ""} />
          </div>
        )}
      </div>
    );
  }

  // 🧪 QUIZ (🔥 FIXED — NO STRICT CHECK)
  if (msg.type === "quiz") {
    return (
      <div key={index} className={`message ${msg.sender}`}>
        <Quiz text={msg.text} />
      </div>
    );
  }

  // 📚 STUDY
  if (msg.type === "study") {
    return (
      <div key={index} className={`message ${msg.sender}`}>
        <StudyMaterial text={msg.text} />
      </div>
    );
  }

  // 💬 DEFAULT TEXT
  return (
    <div key={index} className={`message ${msg.sender}`}>
      <div style={{ whiteSpace: "pre-wrap" }}>
        {msg.text}
      </div>
    </div>
  );
})}
            {loading && <div className="message ai">⏳ Thinking...</div>}
            <div ref={chatEndRef}></div>
          </div>

          {/* INPUT */}
          <div className="input-area">
            <input
              type="text"
              placeholder={
                activeTab === "study"
                  ? "Enter topic for study plan..."
                  : "Type here..."
              }
              value={input}
              onChange={(e) => setInput(e.target.value)}
            />

            <div className="actions">
              {activeTab === "chat" && (
                <>
                  <button onClick={askChat}>Ask</button>
                  <button onClick={summarize}>Summary</button>
                  <button onClick={mindmap}>Mindmap</button>
                  <button onClick={getStudyMaterial}>📚 Study Material</button>
                </>
              )}

              {activeTab === "notes" && (
                <>
                  <button onClick={askNotes}>Ask PDF</button>
                  <button onClick={summarizePDF}>PDF Summary</button>
                  <button onClick={mindmapPDF}>PDF Mindmap</button>
                </>
              )}

              {activeTab === "quiz" && (
                <>
                  <button onClick={generateQuizPDF}>📄 Quiz from PDF</button>
                  <button onClick={generateQuizTopic}>🧠 Quiz from Topic</button>
                </>
              )}
            </div>

            {activeTab === "notes" && (
              <div className="upload">
                <input type="file" onChange={handleUpload} />
              </div>
            )}

            {activeTab === "tutor" && (
              <button onClick={askTutor}>Ask Tutor</button>
            )}

            {activeTab === "study" && (
              <div className="study-mode">
                <h2>📚 Study Mode</h2>

                <p style={{ fontSize: "14px", color: "#aaa" }}>
                  🎯 Get a complete roadmap to master any topic
                </p>

                <select
                  onChange={(e) => setLevel(e.target.value)}
                  value={level}
                >
                  <option value="beginner">Beginner</option>
                  <option value="intermediate">Intermediate</option>
                  <option value="advanced">Advanced</option>
                </select>

                <button onClick={getStudyMaterial}>
                  🚀 Generate Study Plan
                </button>
              </div>
            )}
          </div>

        </div>
      </>
    )}

  </div>
);
}

export default App;