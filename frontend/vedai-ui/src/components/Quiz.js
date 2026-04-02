import React, { useState, useEffect } from "react";
import "./Quiz.css";

function Quiz({ text }) {
  const [questions, setQuestions] = useState([]);
  const [answers, setAnswers] = useState([]);
  const [current, setCurrent] = useState(0);
  const [selected, setSelected] = useState({});
  const [submitted, setSubmitted] = useState(false);
  const [feedback, setFeedback] = useState("");
  const [loadingFeedback, setLoadingFeedback] = useState(false);

  // ===============================
  // 🧠 PARSE QUIZ TEXT
  // ===============================
  useEffect(() => {
    if (!text) return;

    try {
      let qPart = text;
      let aPart = "";

      if (text.includes("ANSWERS")) {
        const split = text.split("ANSWERS");
        qPart = split[0];
        aPart = split[1];
      }

      // QUESTIONS
      const qBlocks = qPart.split(/Q\d+[:.]/).slice(1);

      const parsedQuestions = qBlocks.map((block, index) => {
        const lines = block.trim().split("\n").filter(l => l.trim());

        return {
          id: index,
          question: lines[0],
          options: lines.slice(1, 5).map(opt =>
            opt.replace(/^[A-D]\.\s*/, "").trim()
          )
        };
      });

      // ANSWERS
      let parsedAnswers = [];
      if (aPart.includes("[")) {
        const jsonMatch = aPart.match(/\[([\s\S]*)\]/);
        if (jsonMatch) {
          parsedAnswers = JSON.parse(`[${jsonMatch[1]}]`);
        }
      }

      setQuestions(parsedQuestions);
      setAnswers(parsedAnswers);

    } catch (err) {
      console.error("Quiz Parse Error:", err);
    }
  }, [text]);

  // ===============================
  // 🎯 SELECT OPTION
  // ===============================
  const handleSelect = (qIndex, option) => {
    if (submitted) return;

    setSelected(prev => ({
      ...prev,
      [qIndex]: option
    }));
  };

  // ===============================
  // 🚀 SUBMIT
  // ===============================
  const handleSubmit = () => {
    setSubmitted(true);
    getAIFeedback();
  };

  // ===============================
  // 📊 SCORE
  // ===============================
  const score = answers.reduce((acc, ans, i) => {
    return acc + (selected[i] === ans.correct ? 1 : 0);
  }, 0);

  if (!questions.length) return <div>Loading Quiz...</div>;

  const q = questions[current];

  // ===============================
  // 🤖 AI FEEDBACK
  // ===============================
  const getAIFeedback = async () => {
    setLoadingFeedback(true);

    try {
      const res = await fetch("http://127.0.0.1:8000/quiz-feedback", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          questions: questions.map(q => q.question),
          answers: questions.map((_, i) => selected[i] || "Not Answered")
        })
      });

      const data = await res.json();
      setFeedback(data.feedback);

    } catch (err) {
      console.error(err);
      setFeedback("❌ Failed to get AI feedback");
    }

    setLoadingFeedback(false);
  };

  // ===============================
  // UI
  // ===============================
  return (
    <div className="quiz-container">

      {/* CARD */}
      <div className="quiz-card">
        <h3>Question {current + 1} / {questions.length}</h3>
        <p>{q.question}</p>

        {/* OPTIONS */}
        <div className="options">
          {q.options.map((opt, i) => {
            const optionLetter = ["A", "B", "C", "D"][i];
            const correctAnswer = answers[current]?.correct;

            let className = "option";

            if (submitted) {
              if (optionLetter === correctAnswer) className += " correct";
              else if (selected[current] === optionLetter) className += " wrong";
              className += " disabled";
            } else if (selected[current] === optionLetter) {
              className += " selected";
            }

            return (
              <div
                key={i}
                className={className}
                onClick={() => handleSelect(current, optionLetter)}
              >
                <strong>{optionLetter}.</strong> {opt}
              </div>
            );
          })}
        </div>

        {/* NAVIGATION */}
        <div className="quiz-nav">
          <button
            disabled={current === 0}
            onClick={() => setCurrent(current - 1)}
          >
            ⬅ Prev
          </button>

          {current < questions.length - 1 ? (
            <button onClick={() => setCurrent(current + 1)}>
              Next ➡
            </button>
          ) : (
            <button onClick={handleSubmit}>
              🚀 Submit
            </button>
          )}
        </div>
      </div>

      {/* RESULT */}
      {submitted && (
        <div className="quiz-result">
          <h2>🏆 Score: {score} / {questions.length}</h2>

          <div className="explanations">
            {answers.map((ans, i) => (
              <div key={i} className="exp-card">
                <p><strong>Q{i + 1}</strong></p>
                <p>Correct: {ans.correct}</p>
                <p>{ans.explanation}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* LOADING */}
      {loadingFeedback && (
        <div className="quiz-loading">
          🤖 Analyzing your performance...
        </div>
      )}

      {/* AI FEEDBACK */}
      {feedback && (
        <div className="ai-feedback">
          <h3>🤖 AI Feedback</h3>
          <div style={{ whiteSpace: "pre-wrap" }}>
            {feedback}
          </div>
        </div>
      )}

    </div>
  );
}

export default Quiz;