import React,{useEffect} from "react";
import "./HomePage.css";

function HomePage({ setActiveTab }) {

  useEffect(() => {
  const move = (e) => {
    document.body.style.setProperty("--x", e.clientX + "px");
    document.body.style.setProperty("--y", e.clientY + "px");
  };

  window.addEventListener("mousemove", move);
  return () => window.removeEventListener("mousemove", move);
}, []);

  return (
    <div className="home">

      {/* NAVBAR */}
      <header className="home-navbar">
        <div className="logo">
         <img src="/logo.png" alt="VedAI" width="30" style={{ marginRight: "8px" }} />
         VedAI
        </div>
        <nav>
          <button onClick={() => setActiveTab("chat")}>Chat</button>
          <button onClick={() => setActiveTab("notes")}>Notes</button>
          <button onClick={() => setActiveTab("quiz")}>Quiz</button>
          <button onClick={() => setActiveTab("study")}>Study</button>
        </nav>
      </header>

      {/* HERO */}
      <section className="hero">
        <div className="hero-text">
          <h1 className="hero-title">AI-Powered Study Assistant</h1>
          <p className="hero-subtitle">
            Learn faster with smart notes, quizzes, mindmaps and AI tutor.
          </p>

          <div className="hero-buttons">
            <button onClick={() => setActiveTab("chat")}>
              🚀 Get Started
            </button>

            <button onClick={() => setActiveTab("study")}>
              📚 Explore Study Mode
            </button>
          </div>
        </div>

        <div className="hero-img">
          <img
            src="https://cdn-icons-png.flaticon.com/512/4712/4712109.png"
            alt="AI"
          />
        </div>
      </section>

      {/* FEATURES */}
      <section className="features">
        <div className="feature">
          <h3>🧠 AI Tutor</h3>
          <p>Clear concepts like a real teacher</p>
        </div>

        <div className="feature">
          <h3>📄 Smart Notes</h3>
          <p>Ask questions from your PDFs instantly</p>
        </div>

        <div className="feature">
          <h3>🧪 Quiz Generator</h3>
          <p>Test your knowledge automatically</p>
        </div>

        <div className="feature">
          <h3>📊 Mindmaps</h3>
          <p>Visual learning made easy</p>
        </div>
      </section>

      {/* CTA */}
      <section className="cta">
        <h2>Start learning smarter with VedAI</h2>
        <button onClick={() => setActiveTab("chat")}>
          Start Now
        </button>
      </section>

      {/* FOOTER */}
<div className="footer">
  <div className="footer-content">
    <h2>VedAI</h2>
    <p>AI-powered study assistant for smarter learning 🚀</p>

    <div className="footer-links">
       <p>📧 Email: harshadbandiwar@gmail.com</p>
       <p>💬 Feedback welcome!</p>
    </div>
    <p className="copyright">
      © {new Date().getFullYear()} VedAI. All rights reserved.
    </p>
  </div>
</div>

    </div>
  );
}

export default HomePage;