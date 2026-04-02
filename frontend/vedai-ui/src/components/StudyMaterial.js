import React from "react";

function StudyMaterial({ text }) {
  if (!text) return null;

  const lines = text.split("\n");

  return (
    
    <div className="study-container">

      {lines.map((line, i) => {
        let clean = line.trim();
        if (!clean) return null;

        // ===============================
        // 🎯 HEADINGS (detect smartly)
        // ===============================
        if (
          clean.startsWith("**") ||
          clean.toLowerCase().includes("topics") ||
          clean.toLowerCase().includes("websites") ||
          clean.toLowerCase().includes("youtube") ||
          clean.toLowerCase().includes("books") ||
          clean.toLowerCase().includes("roadmap")
        ) {
          return (
            <h3
              key={i}
              style={{
                marginTop: "22px",
                marginBottom: "8px",
                color: "#ff9800"
              }}
            >
              {clean.replace(/\*\*/g, "")}
            </h3>
          );
        }

        // ===============================
        // 🔗 EXTRACT URL
        // ===============================
        const urlMatch = clean.match(/https?:\/\/[^\s]+/);

        if (urlMatch) {
          const url = urlMatch[0];

          // 🎥 YouTube (search or video)
          if (url.includes("youtube.com") || url.includes("youtu.be")) {
            return (
              <p key={i}>
                ▶️{" "}
                <a
                  href={url}
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{ color: "#ff4d4d" }}
                >
                  Watch on YouTube
                </a>
              </p>
            );
          }

          // 🌐 Normal link
          return (
            <p key={i}>
              🌐{" "}
              <a
                href={url}
                target="_blank"
                rel="noopener noreferrer"
                style={{ color: "#4da6ff" }}
              >
                {url}
              </a>
            </p>
          );
        }

        // ===============================
        // 📄 CLEAN TEXT
        // ===============================
        clean = clean
          .replace(/^\d+\.\s*/, "")   // remove 1. 2.
          .replace(/^[-•]\s*/, "")    // remove bullets
          .replace(/\*\*/g, "");      // remove **

        return <p key={i}>• {clean}</p>;
      })}


    </div>
  );
}

export default StudyMaterial;