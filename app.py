from flask import Flask, render_template, request, jsonify
import os
import string
import nltk
from dotenv import load_dotenv
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import google.generativeai as genai

# =============================
# ENV + GEMINI
# =============================
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY missing in .env")

genai.configure(api_key=API_KEY)

model = genai.GenerativeModel("models/gemini-2.5-flash")

# =============================
# FLASK
# =============================
app = Flask(__name__)

# =============================
# NLTK
# =============================
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

stemmer = SnowballStemmer("french")
STOP_WORDS = set(stopwords.words("french"))

# =============================
# BASE KNOWLEDGE
# =============================
BASE = [
    {
        "categorie": "Démarrage",
        "mots_cles": ["demarr", "allum", "boot", "bip", "noir"],
        "reponse": "Vérifiez alimentation et câble."
    },
    {
        "categorie": "Lenteur",
        "mots_cles": ["lent", "freeze", "lag", "ramer"],
        "reponse": "Fermez les programmes inutiles."
    },
    {
        "categorie": "Internet",
        "mots_cles": ["wifi", "internet", "reseau", "connexion"],
        "reponse": "Redémarrez le routeur."
    },
    {
        "categorie": "Virus",
        "mots_cles": ["virus", "malware", "trojan"],
        "reponse": "Lancez un antivirus complet."
    }
]

# =============================
# MEMORY
# =============================
sessions = {}

# =============================
# NLP
# =============================
def pretraiter(text):
    text = text.lower()

    accents = {
        'é': 'e','è': 'e','ê': 'e','ë': 'e',
        'à': 'a','â': 'a','ä': 'a',
        'ô': 'o','ö': 'o',
        'û': 'u','ù': 'u','ü': 'u',
        'î': 'i','ï': 'i',
        'ç': 'c'
    }

    for a, b in accents.items():
        text = text.replace(a, b)

    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)

    return [
        stemmer.stem(t)
        for t in tokens
        if t not in STOP_WORDS and len(t) > 2
    ]

# =============================
# BASE MATCH
# =============================
def trouver_reponse(message):
    words = pretraiter(message)

    best = None
    best_score = 0

    for cat in BASE:
        score = sum(1 for w in words if w in cat["mots_cles"])
        if score > best_score:
            best_score = score
            best = cat

    return best["reponse"] if best else None

# =============================
# GEMINI AI
# =============================
def ai_response(user_msg, base_response, history):
    try:
        history_text = ""
        for msg in history[-5:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"

        prompt = f"""
You are a computer technician.

History:
{history_text}

User problem:
{user_msg}

Base solution:
{base_response}

Rules:
- simple explanation
- step by step
- max 5 lines
- ask one question
"""

        response = model.generate_content(prompt)
        return response.text[:700]

    except Exception as e:
        print("AI ERROR:", e)
        return "Erreur IA, réessayez plus tard."

# =============================
# ROUTES
# =============================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message", "").strip()

    user_id = request.remote_addr

    if user_id not in sessions:
        sessions[user_id] = []

    history = sessions[user_id]

    base_response = trouver_reponse(message)

    history.append({"role": "user", "content": message})

    final_response = ai_response(message, base_response, history)

    history.append({"role": "assistant", "content": final_response})

    return jsonify({"reponse": final_response})

# =============================
# RUN
# =============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)