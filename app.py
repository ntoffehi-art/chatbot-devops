from flask import Flask, render_template, request, jsonify
import string
import nltk
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gpt4all import GPT4All

# =============================
# GPT4ALL MODEL
# =============================
# Après — chemin complet direct
llm = GPT4All(
    model_name="Llama-3.2-1B-Instruct-Q4_0.gguf",
    model_path=r"C:\Users\dell\AppData\Local\nomic.ai\GPT4All\models",
    allow_download=False
)
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
# MEMORY
# =============================
sessions = {}

# =============================
# BASE KNOWLEDGE
# =============================
BASE = [
    {
        "categorie": "Demarrage",
        "mots_cles": ["demarr", "boot", "noir", "bip", "demarre"],
        "solutions": [
            "Vérifiez que le câble d'alimentation est bien branché",
            "Essayez une autre prise électrique",
            "Écoutez s'il y a des bips au démarrage",
            "Vérifiez si l'écran s'allume"
        ],
        "question": "Est-ce que votre PC fait un bruit ou affiche quelque chose ?"
    },
    {
        "categorie": "Internet",
        "mots_cles": ["wifi", "internet", "reseau", "connexion"],
        "solutions": [
            "Vérifiez que le WiFi est activé",
            "Redémarrez votre routeur",
            "Essayez avec un autre appareil",
            "Vérifiez les câbles réseau"
        ],
        "question": "Le problème concerne tous les appareils ou seulement votre PC ?"
    }
]

# =============================
# GREETING
# =============================
def is_greeting(message):
    greetings = ["bonjour", "salut", "hello", "salam", "hi"]
    return any(g in message.lower() for g in greetings)

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
# CATEGORY DETECTION
# =============================
def trouver_categorie(message):
    words = set(pretraiter(message))

    best_cat = None
    best_score = 0

    for cat in BASE:
        keywords = set(cat["mots_cles"])
        score = len(words.intersection(keywords))

        if score > best_score:
            best_score = score
            best_cat = cat

    if best_score == 0:
        return None

    return best_cat

# =============================
# RESPONSE BUILDER
# =============================
def build_response(cat):
    steps = "\n".join([f"{i+1}. {s}" for i, s in enumerate(cat["solutions"])])
    return f"Voici les étapes à suivre :\n{steps}\n\n{cat['question']}"

# =============================
# GPT4ALL AI
# =============================
def ai_response(text):
    try:
        prompt = f"""
Tu es un assistant technique.
Rends ce texte clair, simple et naturel en français.

{text}
"""
        with llm.chat_session():
            response = llm.generate(prompt, max_tokens=200)

        return response.strip()

    except Exception as e:
        print("Erreur AI:", e)
        return text

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

    if is_greeting(message):
        return jsonify({"reponse": "Bonjour 👋 Décrivez votre problème."})

    cat = trouver_categorie(message)

    if not cat:
        reply = "Je ne comprends pas bien. Pouvez-vous préciser votre problème ?"
    else:
        base = build_response(cat)
        reply = ai_response(base)

    return jsonify({"reponse": reply})

# =============================
# RUN
# =============================
if __name__ == "__main__":
    print("Démarrage du serveur...")
    try:
        app.run(host="0.0.0.0", port=5000, debug=False)
    except Exception as e:
        print("Erreur serveur:", e)