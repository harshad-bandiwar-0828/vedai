from groq import Groq

# ==============================
# 1. SET YOUR API KEY
# ==============================
import os
groq_api_key = os.getenv("GROQ_API_KEY")


# ==============================
# 2. SUMMARIZE FUNCTION
# ==============================
def summarize_text(text):

    prompt = f"""
Summarize the following text into 5 short bullet points.

Text:
{text}

Summary:
"""

    response = groq_api_key.chat.completions.create(
        model="llama-3.1-8b-instant",
       messages=[
    {"role": "system", "content": "You are VedAI. Answer ONLY from notes."},
    {"role": "user", "content": prompt}
]
    )

    return response.choices[0].message.content.strip()


# ==============================
# 3. TEST
# ==============================
if __name__ == "__main__":

    print("📌 VedAI Summarizer")
    print("-------------------")

    text = input("\nEnter text to summarize:\n")

    result = summarize_text(text)

    print("\n📝 Summary:\n")
    print(result)