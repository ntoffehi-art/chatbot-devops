from flask import Flask, request, jsonify, render_template
from openai import OpenAI
from dotenv import load_dotenv
import os

# =========================
# LOAD ENV
# =========================
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("❌ GROQ_API_KEY manquante dans .env")

# =========================
# CLIENT GROQ
# =========================
client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

# =========================
# FLASK APP
# =========================
app = Flask(_name_)

# Historique en mémoire : { user_id: [{ role, content }, ...] }
historiques = {}

SYSTEM_PROMPT = """Tu es TechBot, un assistant expert UNIQUEMENT en pannes informatiques.
Règles strictes :
- Refuse poliment tout sujet hors informatique (cuisine, politique, etc.)
- Donne des solutions claires, numérotées étape par étape
- Utilise des emojis pour rendre tes réponses lisibles
- Si la solution a plusieurs étapes, numérote-les (1️⃣, 2️⃣, 3️⃣...)
- Sois concis mais complet
- Si tu ne sais pas, dis-le honnêtement
- Si l'utilisateur dit que c'est résolu, félicite-le chaleureusement"""

MAX_HISTORY = 20  # Garde les 20 derniers messages pour ne pas dépasser les tokens

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"reponse": "❌ Données invalides reçues."})

        message = data.get("message", "").strip()
        user_id = data.get("user_id", "anonymous")

        if not message:
            return jsonify({"reponse": "⚠️ Message vide, veuillez écrire quelque chose."})

        # Initialiser l'historique pour ce user si nécessaire
        if user_id not in historiques:
            historiques[user_id] = []

        # Ajouter le message de l'utilisateur à l'historique
        historiques[user_id].append({
            "role": "user",
            "content": message
        })

        # Limiter la taille de l'historique pour éviter de dépasser les tokens
        if len(historiques[user_id]) > MAX_HISTORY:
            historiques[user_id] = historiques[user_id][-MAX_HISTORY:]

        # Construire les messages à envoyer (system + historique complet)
        messages_api = [{"role": "system", "content": SYSTEM_PROMPT}] + historiques[user_id]

        # Appel à l'AI 
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages_api,
            temperature=0.5,
            max_tokens=500
        )

        bot_reply = response.choices[0].message.content

        # Ajouter la réponse du bot à l'historique
        historiques[user_id].append({
            "role": "assistant",
            "content": bot_reply
        })

        return jsonify({"reponse": bot_reply})

    except Exception as e:
        print(f"❌ ERREUR : {e}")
        return jsonify({"reponse": "⚠️ Erreur serveur. Vérifiez votre clé API et réessayez."})

@app.route("/reset", methods=["POST"])
def reset():
    data = request.get_json()
    user_id = data.get("user_id", "anonymous")

    if user_id in historiques:
        del historiques[user_id]

    return jsonify({"status": "ok"})
if _name_ == "_main_":
    print("🚀 TechBot lancé sur http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
