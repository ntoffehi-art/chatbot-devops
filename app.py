from flask import Flask, render_template, request, jsonify
import nltk
import string
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Téléchargement des ressources NLTK
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

app = Flask(__name__)

# ============================================================
#   BASE DE CONNAISSANCES
# ============================================================

BASE = [
    {
        "categorie": "Démarrage",
        "mots_cles": ["démarrer", "démarrage", "allumer", "allumage", "boot", "bip", "noir", "rien", "éteint"],
        "reponse": "💡 PC ne démarre pas :\n1. Vérifiez le câble d'alimentation\n2. Vérifiez la prise électrique\n3. Appuyez 10 secondes sur le bouton power\n4. Débranchez tous les périphériques USB\n5. Si vous entendez des bips → notez leur nombre\n⚠️ Problème possible : alimentation ou carte mère."
    },
    {
        "categorie": "Écran bleu",
        "mots_cles": ["bleu", "bsod", "plantage", "redémarrage", "crash", "blue screen", "erreur système"],
        "reponse": "🔵 Écran bleu (BSOD) :\n1. Notez le code d'erreur affiché\n2. Redémarrez en mode sans échec (F8)\n3. Désinstallez les derniers drivers\n4. Lancez : sfc /scannow\n5. Vérifiez la RAM avec l'outil de diagnostic\n⚠️ Sauvegardez vos données immédiatement."
    },
    {
        "categorie": "Lenteur",
        "mots_cles": ["lent", "lenteur", "freeze", "geler", "bloquer", "ramer", "lag", "performance", "ralentir"],
        "reponse": "🐢 PC lent ou qui freeze :\n1. Ouvrez le Gestionnaire des tâches (Ctrl+Shift+Échap)\n2. Identifiez le processus qui consomme le plus\n3. Supprimez les fichiers temporaires (%temp%)\n4. Désactivez les programmes au démarrage\n5. Vérifiez si le disque est presque plein\n⚠️ RAM toujours à 90%+ → envisagez une mise à niveau."
    },
    {
        "categorie": "Internet",
        "mots_cles": ["internet", "wifi", "réseau", "connexion", "navigateur", "site", "web", "dns", "ip", "câble"],
        "reponse": "🌐 Problème Internet :\n1. Redémarrez le routeur (débranchez 30 secondes)\n2. Vérifiez si d'autres appareils ont internet\n3. Tapez : ping google.com dans le terminal\n4. Essayez : ipconfig /release puis ipconfig /renew\n5. Désactivez puis réactivez la carte réseau\n⚠️ Seul votre PC sans internet → problème côté PC."
    },
    {
        "categorie": "Virus",
        "mots_cles": ["virus", "malware", "infecté", "piraté", "antivirus", "pub", "ransomware", "trojan", "sécurité"],
        "reponse": "🦠 Virus ou Malware :\n1. Déconnectez le PC d'internet immédiatement\n2. Lancez Windows Defender (analyse complète)\n3. Téléchargez Malwarebytes pour un scan\n4. Supprimez les extensions suspectes du navigateur\n5. Changez vos mots de passe depuis un autre appareil\n⚠️ Ne payez JAMAIS une rançon."
    },
    {
        "categorie": "Périphériques",
        "mots_cles": ["clavier", "souris", "périphérique", "usb", "brancher", "reconnaitre", "driver", "pilote"],
        "reponse": "⌨️ Clavier ou souris :\n1. Débranchez et rebranchez le périphérique\n2. Essayez un autre port USB\n3. Testez le périphérique sur un autre PC\n4. Vérifiez le Gestionnaire de périphériques\n5. Pour sans-fil → changez les piles\n⚠️ Problème sur tous les ports → carte mère."
    },
    {
        "categorie": "Son",
        "mots_cles": ["son", "audio", "muet", "haut-parleur", "casque", "volume", "silence"],
        "reponse": "🔇 Pas de son :\n1. Vérifiez que le son n'est pas en sourdine\n2. Clic droit icône son → Périphériques de lecture\n3. Vérifiez le bon périphérique par défaut\n4. Mettez à jour les drivers audio\n5. Redémarrez le service audio (services.msc)"
    },
    {
        "categorie": "Imprimante",
        "mots_cles": ["imprimante", "imprimer", "impression", "encre", "papier", "bourrage", "printer"],
        "reponse": "🖨️ Imprimante :\n1. Vérifiez que l'imprimante est allumée et connectée\n2. Annulez tous les travaux en attente\n3. Redémarrez le Spouleur d'impression (services.msc)\n4. Réinstallez les drivers\n5. Vérifiez le niveau d'encre et le papier"
    },
]

# ============================================================
#   MOTEUR NLP
# ============================================================

stemmer = SnowballStemmer("french")
#nettoyage
STOP_WORDS = set(stopwords.words("french"))

def pretraiter(texte):
    texte = texte.lower()
    accents = {
        'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
        'à': 'a', 'â': 'a', 'ä': 'a',
        'ô': 'o', 'ö': 'o',
        'û': 'u', 'ù': 'u', 'ü': 'u',
        'î': 'i', 'ï': 'i',
        'ç': 'c'
    }
    for accent, lettre in accents.items():
        texte = texte.replace(accent, lettre)

    #supprimes les symboles
    texte = texte.translate(str.maketrans("", "", string.punctuation))
    #coupe la chine en une liste de mots
    tokens = word_tokenize(texte, language="french")
    racines = []
    for mot in tokens:
        if mot not in STOP_WORDS and len(mot) > 2:
            #ajout ce res à liste finale + reduit le mots à liste finale
            racines.append(stemmer.stem(mot))
    return racines

def trouver_reponse(message):
    racines = pretraiter(message)
    if not racines:
        return None

    meilleur_score = 0
    meilleure_reponse = None

    for categorie in BASE:
        racines_cat = [stemmer.stem(m.lower()) for m in categorie["mots_cles"]]
        score = len(set(racines) & set(racines_cat))
        if score > meilleur_score:
            meilleur_score = score
            meilleure_reponse = categorie["reponse"]

    return meilleure_reponse if meilleur_score > 0 else None

# ============================================================
#   ROUTES FLASK
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()#prend le message de l'utilisateur
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"reponse": "Veuillez écrire votre problème."})

    salutations = ["bonjour", "salam", "salut", "bonsoir", "hello"]
    if any(mot in message.lower() for mot in salutations):
        return jsonify({"reponse": "Bonjour ! Décrivez votre panne informatique et je vais vous aider. 😊"})

    reponse = trouver_reponse(message)

    if reponse:
        return jsonify({"reponse": reponse})
    else:
        return jsonify({"reponse": "Je n'ai pas compris votre problème. Essayez de le décrire autrement (ex: 'PC lent', 'pas internet', 'écran bleu')."})

if __name__ == "__main__":
    app.run(debug=True)
