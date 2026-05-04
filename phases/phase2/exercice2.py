# ─────────────────────────────────────────
# Exercice 2 — Output structuré JSON
# ─────────────────────────────────────────
"""
Objectif : forcer le modèle à répondre dans un format précis et parser
le résultat directement en Python.

Ce que fait ce code :
- Demande une analyse de texte au format JSON strict
- Parse la réponse avec json.loads()
- Affiche chaque champ séparément

Pour jouer :
- Change le format JSON demandé (ajoute un champ "langue", "longueur", etc.)
- Teste avec un texte ambigu et observe comment le modèle gère l'incertitude
- Remplace json.loads() par une gestion d'erreur robuste — le modèle ne respecte
  pas toujours le format, c'est un problème réel en production
- Essaie sans préciser "UNIQUEMENT un JSON" et vois ce qui se passe
"""

import ollama
import json

MODEL = "llama3.2:3b"

texte = "Livraison en 24h, produit conforme à la description, je suis très satisfait."

prompt = f"""
Analyse ce texte et retourne UNIQUEMENT un JSON valide, sans texte autour :
{{
  "sentiment": "positif" | "négatif" | "neutre",
  "score": 0.0 à 1.0,
  "mots_cles": ["mot1", "mot2"]
}}

Texte : "{texte}"
"""

response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
raw = response["message"]["content"]

try:
    data = json.loads(raw)
    print(f"Sentiment : {data['sentiment']}")
    print(f"Score     : {data['score']}")
    print(f"Mots clés : {data['mots_cles']}")
except json.JSONDecodeError:
    print("Le modèle n'a pas respecté le format JSON :")
    print(raw)
