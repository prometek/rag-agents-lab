# ─────────────────────────────────────────
# Exercice 4 — System prompt
# ─────────────────────────────────────────
"""
Objectif : comprendre comment le system prompt contrôle le comportement global
du modèle — ton, contraintes, format, personnalité.

Ce que fait ce code :
- Définit un assistant spécialisé via le system prompt
- Teste une question dans son domaine
- Teste une question hors domaine pour voir comment il réagit

Pour jouer :
- Change le domaine (assistant running, assistant cuisine, assistant code)
- Durcis les contraintes ("réponds en maximum 2 phrases", "utilise toujours des bullet points")
- Essaie de "casser" le system prompt avec des questions hors domaine
- Ajoute un ton spécifique ("tu es sarcastique", "tu parles comme un pirate")
"""

import ollama

MODEL = "llama3.2:3b"

system = """
Tu es un assistant financier expert en ETF et gestion de portefeuille.
Tu réponds uniquement sur des sujets financiers.
Si on te pose une question hors sujet, tu réponds poliment que ce n'est pas ton domaine.
Tes réponses sont concises, max 3 phrases.
"""

questions = [
    "C'est quoi la différence entre un ETF et un fonds actif ?",
    "Donne-moi une recette de tiramisu.",
]

for question in questions:
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
    )
    print(f"Q: {question}")
    print(f"R: {response['message']['content']}\n")
