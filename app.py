from flask import Flask, request, jsonify, render_template
from llama_cpp import Llama

app = Flask(__name__)

llm = Llama(
    model_path=r"C:\Users\dell\AppData\Local\nomic.ai\GPT4All\models\Llama-3.2-3B-Instruct-Q4_0.gguf",
    n_ctx=1024,
    n_threads=4,
    verbose=False
)

# ================= NORMALIZE =================
def normalize(text):
    text = text.lower()
    replacements = {
        "é": "e","è": "e","ê": "e",
        "à": "a","â": "a",
        "ù": "u","û": "u",
        "î": "i","ï": "i"
    }
    for a,b in replacements.items():
        text = text.replace(a,b)
    return text

# ================= INTENT =================
def detect_intent(msg):
    msg = normalize(msg)

    if any(w in msg for w in ["internet","wifi","connexion","reseau","net"]):
        return "internet"
    if any(w in msg for w in ["bleu","bsod"]):
        return "bsod"
    if any(w in msg for w in ["lent","lag","ralenti"]):
        return "slow"
    if any(w in msg for w in ["virus","malware"]):
        return "virus"
    if any(w in msg for w in ["clavier","souris"]):
        return "input"
    if any(w in msg for w in ["son","audio"]):
        return "sound"
    if any(w in msg for w in ["imprimante"]):
        return "printer"
    if any(w in msg for w in ["demarre","boot","allume pas"]):
        return "boot"
    if any(w in msg for w in ["c'est resolu", "merci c'est", "parfait", "super"]):
        return "resolved"

    if any(w in msg for w in ["pas encore resolu", "pas resolu", "besoin d'aide"]):
        return "need_help"

    if any(w in msg for w in ["autre probleme", "autre"]):
        return "restart"
    return "unknown"

# ================= RESPONSES =================
def get_response(intent):
    base = {
        "boot": "⚠️ PC ne démarre pas\n1. Vérifie que le câble d'alimentation est bien branché\n2. Teste avec un autre écran ou câble HDMI\n3. Retire les barrettes RAM et remets-les\n4. Écoute les bips au démarrage\n👉 Si rien ne marche : carte mère ou alimentation défaillante",
        "internet": "🌐 Internet ne marche pas\n1. Redémarre le routeur (débranche 30 sec)\n2. Vérifie si le WiFi est activé\n3. Teste avec un autre appareil\n4. Oublie le réseau WiFi et reconnecte-toi",
        "bsod": "🔵 Écran bleu (BSOD)\n1. Note le code d'erreur affiché\n2. Mets à jour tes pilotes (carte graphique surtout)\n3. Vérifie la RAM avec l'outil mdsched (Win+R)\n4. Vérifie les mises à jour Windows récentes",
        "slow": "🐢 PC lent\n1. Ouvre le gestionnaire des tâches (Ctrl+Shift+Esc)\n2. Vérifie CPU/RAM — ferme les programmes lourds\n3. Désactive les programmes au démarrage\n4. Libère de l'espace disque (min 15% libre)",
        "virus": "🦠 Virus / Malware\n1. Lance Windows Defender → Analyse complète\n2. Télécharge Malwarebytes (gratuit) et analyse\n3. Vérifie les extensions suspectes dans ton navigateur\n4. Change tes mots de passe après nettoyage",
        "input": "⌨️ Clavier / Souris\n1. Débranche et rebranche le câble USB\n2. Essaie un autre port USB\n3. Pour Bluetooth : désactive et réactive\n4. Teste le périphérique sur un autre PC",
        "sound": "🔇 Problème de son\n1. Vérifie que le volume n'est pas coupé (icône en bas à droite)\n2. Clic droit sur l'icône son → Résoudre les problèmes\n3. Vérifie le bon périphérique de sortie sélectionné\n4. Mets à jour les pilotes audio dans le gestionnaire de périphériques",
        "printer": "🖨️ Imprimante\n1. Vérifie que l'imprimante est allumée et connectée\n2. Supprime les tâches bloquées dans la file d'impression\n3. Redémarre le spouleur : Win+R → services.msc → Spouleur d'impression\n4. Réinstalle les pilotes depuis le site du fabricant",
        "resolved": "✅ Parfait ! Ravi d'avoir aidé.\n💬 Si tu as un autre problème, décris-le !",
        "restart":  "🔄 Pas de problème ! Quel est ton nouveau problème ?\n(internet, virus, écran bleu...)",
        "need_help": "🔧 D'accord, continuons le diagnostic.\nDécris-moi mieux le problème ou essaie de redémarrer le PC.",
    }

    # Pas de "💡 Essaie aussi" pour ces intents
    no_suffix = ["resolved", "restart", "need_help"]

    if intent in base:
        if intent in no_suffix:
            return base[intent]
        return base[intent] + "\n\n💡 Essaie aussi : internet, écran bleu, lent..."
    return None

def reformuler(texte):
    try:
        res = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "Tu es TechBot. Reformule ces étapes en français simple et amical. Garde exactement les mêmes étapes, ne rajoute rien."},
                {"role": "user", "content": texte}
            ],
            max_tokens=200,
            temperature=0.3
        )
        result = res["choices"][0]["message"]["content"].strip()
        if len(result) > 60:
            return result
        return texte
    except:
        return texte

# ================= FALLBACK =================
def fallback_llm(text):
    try:
        res = llm.create_chat_completion(
            messages=[
                {"role":"system","content":"Assistant IT. Réponds court."},
                {"role":"user","content":text}
            ],
            max_tokens=60,
            temperature=0.3
        )
        return res["choices"][0]["message"]["content"]
    except:
        return "🤔 Précise: internet, écran bleu, lent..."

# ================= ROUTES =================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    message = request.json.get("message","")

    intent = detect_intent(message)
    print(f"[DEBUG] {message} → {intent}")

    response = get_response(intent)

    if response:
        return jsonify({"reponse": reformuler(response)})
    return jsonify({"reponse": fallback_llm(message)})

if __name__ == "__main__":
    print("🚀 TechBot prêt sur http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)