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
# MEMORY
# =============================
sessions = {}
# ENV + GEMINI
# =============================
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
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
        "categorie": "Demarrage",
        "mots_cles": ["demarr", "boot", "noir", "bip"],
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
    },
    {
        "categorie": "Lenteur",
        "mots_cles": ["lent", "lag", "freeze", "ramer"],
        "solutions": [
            "Fermez les programmes inutiles",
            "Redémarrez votre ordinateur",
            "Vérifiez l'espace disque",
            "Lancez un scan antivirus"
        ],
        "question": "Votre PC est lent tout le temps ou seulement avec certains programmes ?"
    },
    {
        "categorie": "Imprimante",
        "mots_cles": ["imprimante", "print", "papier", "scan", "scanner"],
        "solutions": [
            "Vérifiez que l’imprimante est bien branchée",
            "Vérifiez les niveaux d’encre",
            "Redémarrez l’imprimante",
            "Vérifiez les pilotes (drivers)"
        ],
        "question": "Est-ce que l’imprimante est reconnue par l’ordinateur ?"
    },
    {
        "categorie": "Virus",
        "mots_cles": ["virus", "malware", "trojan", "infect"],
        "solutions": [
            "Lancez un antivirus complet",
            "Supprimez les programmes suspects",
            "Mettez à jour votre antivirus",
            "Redémarrez en mode sécurisé"
        ],
        "question": "Avez-vous installé un logiciel récemment ?"
    },
    {
        "categorie": "Clavier/Souris",
        "mots_cles": ["clavier", "souris", "mouse", "keyboard"],
        "solutions": [
            "Vérifiez la connexion USB ou Bluetooth",
            "Changez les piles si c’est sans fil",
            "Redémarrez le PC",
            "Essayez un autre port USB"
        ],
        "question": "Est-ce que le périphérique est détecté par le PC ?"
    },
    {
        "categorie": "Son",
        "mots_cles": ["son", "audio", "haut-parleur", "speaker"],
        "solutions": [
            "Vérifiez le volume",
            "Vérifiez les périphériques audio",
            "Mettez à jour les drivers son",
            "Redémarrez le PC"
        ],
        "question": "Le problème est-il sur casque ou haut-parleurs ?"
    }
]

# =============================

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
# DETECTION CATEGORY
# =============================
def trouver_categorie(message):
    words = set(pretraiter(message))
    scores = []
    for cat in BASE:
        keywords = set(cat["mots_cles"])
        score = len(words.intersection(keywords))
        if score > 0:
            confidence = score / len(keywords)
            scores.append({
                "cat": cat,
                "score": score,
                "confidence": confidence
            })
    if not scores:
        return None, None

    scores.sort(key=lambda x: (x["confidence"], x["score"]), reverse=True)

    best = scores[0]["cat"]

    if scores[0]["confidence"] < 0.25:
        return None, None

    if len(scores) > 1:
        second = scores[1]["cat"]

        if abs(scores[0]["score"] - scores[1]["score"]) <= 1:
            return best, second

    return best, None

# RESPONSE BUILD
def build_response(cat):
    steps = "\n".join([f"{i+1}. {s}" for i, s in enumerate(cat["solutions"])])
    return {
        "text": f"Voici les étapes à suivre :\n{steps}",
        "question": cat["question"]
    }
# =============================
# GEMINI AI
# =============================
def ai_response(base_text):
    try:
        prompt = f"""Tu es TechBot, assistant technique.
Améliore ce texte : rends-le clair, naturel, en français.
Ne change pas le sens. Garde la numérotation si présente.
IMPORTANT: Réponds UNIQUEMENT avec le texte amélioré, sans introduction, sans phrase comme "Voici une version améliorée", commence directement par le contenu.

Texte:
{base_text}
"""
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Erreur Gemini: {e}")
        return base_text
    
def is_greeting(message):
    greetings = ["bonjour", "salut", "hello", "salam", "bonsoir", "hi", "hey"]
    return any(g in message.lower() for g in greetings)
# =============================
# =============================
# GEMINI FALLBACK (problème non reconnu)
# =============================
def gemini_fallback(message, history):
    try:
        history_text = "\n".join([
            f"{h['role']}: {h['content']}"
            for h in history[-6:]
        ])

        prompt = f"""
Tu es un assistant technique expert.

Même si ce n'est pas dans ta base, comprends le problème.

Donne:
- diagnostic probable
- solution simple
- une question finale

Historique:
{history_text}

Problème:
{message}
"""
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Erreur Gemini fallback: {e}")
        return "Pouvez-vous préciser votre problème ?"
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
        # 🧠 INIT SESSION PROPERLY
    if len(sessions) > 100:
        sessions.clear()
    if user_id not in sessions:
        sessions[user_id] = {
            "history": [],
            "ambiguity_asked": False,
            "question_asked": False,
            "state": "start"
        }
        
    session = sessions[user_id]
    history = session["history"]
    last_user = history[-2]["content"] if len(history) >= 2 and history[-2]["role"] == "user" else ""
    combined_message = last_user + " " + message

    # 🟢 greeting
    if is_greeting(message):
        return jsonify({
            "reponse": "Bonjour 👋 Je suis TechBot. Décrivez votre problème."
        })

    history.append({"role": "user", "content": message})

    # 🧠 CATEGORY
    cat, second = trouver_categorie(combined_message)
    
    # ❌ NOT UNDERSTOOD
    if not cat:
        reply = gemini_fallback(message, history)
        session["state"] = "fallback"

    # 🤯 AMBIGUITY
    elif second and not session["ambiguity_asked"]:
        reply = f"Est-ce que votre problème concerne plutôt {cat['categorie']} ?"
        session["ambiguity_asked"] = True
        session["state"] = "ambiguity"

    # ✅ NORMAL FLOW
    else:
        base = build_response(cat)
        reply = base["text"]
        # 🔥 only one question
        if not session["question_asked"]:
            reply += "\n\n" + base["question"]
            session["question_asked"] = True
        session["state"] = "solution"
        # AI enhancement ONLY here
        reply = ai_response(reply)

    history.append({"role": "bot", "content": reply})

    return jsonify({"reponse": reply})
# =============================
# RUN
# =============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)