from groq import Groq

# ==============================
# 1. SET API KEY
# ==============================
import os
groq_api_key = os.getenv("GROQ_API_KEY")


# ==============================
# 2. MINDMAP FUNCTION
# ==============================
def generate_mindmap(text):

    prompt = f"""
Convert this into a structured mindmap.

Use format:
Main Topic
  - Subtopic
    - Detail

Text:
{text}

Mindmap:
"""

    response = groq_api_key.chat.completions.create(
        model="llama-3.1-8b-instant",
       messages=[
    {"role": "system", "content": "You are VedAI. You create structured mindmaps."},
    {"role": "user", "content": prompt}
]
    )

    return response.choices[0].message.content.strip()


# ==============================
# 3. RUN FILE (TEST)
# ==============================
if __name__ == "__main__":

    print("🧠 VedAI Mindmap Generator")
    print("--------------------------")

    text = input("\nEnter text:\n")

    result = generate_mindmap(text)

    print("\n🌳 Mindmap:\n")
    print(result)