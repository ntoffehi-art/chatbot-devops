from flask import Flask, render_template, request, jsonify
import nltk
import string
from dotenv import load_dotenv
import os
load_dotenv()
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import google.generativeai as genai

# =============================
# GEMINI SETUP (SECURE)
# =============================
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-pro")

# =============================
# INIT FLASK + NLTK
# =============================
app = Flask(__name__)

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

stemmer = SnowballStemmer("french")
STOP_WORDS = set(stopwords.words("french"))

# =============================
# BASE DE CONNAISSANCES
# =============================
BASE = [
    {
        "categorie": "Démarrage",
        "mots_cles": ["demarr", "allum", "boot", "bip", "noir"],
        "reponse": "Problème possible d’alimentation ou carte mère. Vérifiez câble et prise."
    },
    {
        "categorie": "Lenteur",
        "mots_cles": ["lent", "freeze", "lag", "ramer"],
        "reponse": "PC lent : vérifiez le gestionnaire des tâches et supprimez les programmes inutiles."
    },
    {
        "categorie": "Internet",
        "mots_cles": ["wifi", "internet", "reseau", "connexion"],
        "reponse": "Redémarrez le routeur et testez la connexion avec un autre appareil."
    },
    {
        "categorie": "Virus",
        "mots_cles": ["virus", "malware", "trojan"],
        "reponse": "Déconnectez internet et lancez un scan antivirus complet."
    }
]

# =============================
# MEMOIRE SIMPLE
# =============================
sessions = {}

# =============================
# NLP PREPROCESSING
# =============================
def pretraiter(texte):
    texte = texte.lower()

    accents = {
        'é': 'e','è': 'e','ê': 'e','ë': 'e',
        'à': 'a','â': 'a','ä': 'a',
        'ô': 'o','ö': 'o',
        'û': 'u','ù': 'u','ü': 'u',
        'î': 'i','ï': 'i',
        'ç': 'c'
    }

    for a, b in accents.items():
        texte = texte.replace(a, b)

    texte = texte.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(texte)

    return [
        stemmer.stem(t)
        for t in tokens
        if t not in STOP_WORDS and len(t) > 2
    ]

# =============================
# FIND BASE RESPONSE
# =============================
def trouver_reponse(message):
    racines = pretraiter(message)

    best_score = 0
    best = None

    for cat in BASE:
        score = sum(1 for w in racines if w in cat["mots_cles"])
        if score > best_score:
            best_score = score
            best = cat

    return best["reponse"] if best else None

# =============================
# AI (GEMINI)
# =============================
def ai_response(user_msg, base_response, history):

    history_text = ""
    for msg in history[-5:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    prompt = f"""
Tu es un expert en dépannage informatique.

Historique:
{history_text}

Problème utilisateur:
{user_msg}

Solution technique:
{base_response if base_response else "Aucune solution trouvée"}

Règles:
- Explique simplement
- Étapes claires
- Une seule question à la fin
"""

    response = model.generate_content(prompt)
    return response.text[:800]

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

    if not base_response:
        base_response = "Aucune solution précise trouvée."

    history.append({"role": "user", "content": message})

    final_response = ai_response(message, base_response, history)

    history.append({"role": "assistant", "content": final_response})

    return jsonify({"reponse": final_response})

# =============================
# RUN
# =============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)