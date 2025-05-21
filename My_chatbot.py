
import json
import random
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re

# Load intents
with open("intents.json", "r") as file:
    intents = json.load(file)

# Preprocess data
tags = []
patterns = []

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Train model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)
y = tags

clf = LogisticRegression(random_state=0, max_iter=10000)
clf.fit(X, y)

# Tips & quizzes
biology_tips = [
    "Photosynthesis occurs in the chloroplasts of plant cells.",
    "The cell is the basic unit of life.",
    "Osmosis is the movement of water across a selectively permeable membrane.",
    "Enzymes are biological catalysts that speed up reactions.",
    "DNA carries genetic information in all living organisms."
]

quiz_questions = {
    "What is the basic unit of life?": "The cell",
    "Which organelle is responsible for photosynthesis?": "Chloroplast",
    "What molecule carries genetic information?": "DNA",
    "What is the function of enzymes?": "Speed up chemical reactions",
    "What process moves water through membranes?": "Osmosis"
}

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "quiz_mode" not in st.session_state:
    st.session_state.quiz_mode = False

if "current_question" not in st.session_state:
    st.session_state.current_question = None

# Chatbot
def chatbot_ml(input_text):
    vec = vectorizer.transform([input_text])
    tag = clf.predict(vec)[0]
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "I'm not sure how to respond to that."

def chatbot_pattern(user_input):
    user_tokens = re.findall(r"\w+", user_input.lower())
    for intent in intents["intents"]:
        for pattern in intent['patterns']:
            pattern_tokens = re.findall(r"\w+", pattern.lower())
            if set(pattern_tokens).intersection(user_tokens):
                return random.choice(intent['responses'])
    return chatbot_ml(user_input)

# UI
def main():
    st.set_page_config(page_title="Biology Chatbot", page_icon="🧬", layout="centered")

    st.title("🧬 Welcome to BioBot!")
    st.markdown("**Your friendly biology assistant. Ask me anything from Chapters 1 to 10!**")
    st.info("💡 Tip of the day: " + random.choice(biology_tips))

    st.markdown("### 📖 Select Chapter")
    chapter = st.selectbox("Choose a chapter", ["All", "Chapter 1", "Chapter 2", "Chapter 3", "Chapter 4", "Chapter 5", 
                                                "Chapter 6", "Chapter 7", "Chapter 8", "Chapter 9", "Chapter 10"])

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧪 Start Quiz"):
            st.session_state.quiz_mode = True
            st.session_state.current_question = random.choice(list(quiz_questions.keys()))
        if st.button("❌ End Quiz"):
            st.session_state.quiz_mode = False
            st.session_state.current_question = None

    user_input = st.text_input("You:")

    if user_input:
        if st.session_state.quiz_mode and st.session_state.current_question:
            correct = quiz_questions[st.session_state.current_question]
            if user_input.strip().lower() == correct.lower():
                response = "✅ Correct!"
            else:
                response = f"❌ Incorrect. The correct answer is: {correct}"
            st.session_state.current_question = random.choice(list(quiz_questions.keys()))
        else:
            response = chatbot_pattern(user_input)

        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("bot", response))

    for speaker, message in st.session_state.chat_history:
        if speaker == "user":
            st.markdown(f"<div style='background-color:#DCF8C6;padding:10px;border-radius:10px;margin-bottom:10px'><strong>You:</strong> {message}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background-color:#F1F0F0;padding:10px;border-radius:10px;margin-bottom:10px'><strong>Bot:</strong> {message}</div>", unsafe_allow_html=True)

    if st.session_state.quiz_mode and st.session_state.current_question:
        st.markdown(f"### 🤔 Quiz Question:\n**{st.session_state.current_question}**")

if __name__ == "__main__":
    main()
