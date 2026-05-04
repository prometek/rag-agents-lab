"""
Exercice 1 — Premier appel LLM local

Objectif : comprendre comment un LLM reçoit un message et génère une réponse,
et observer le coût en tokens de chaque échange.

Ce que fait ce code :
- Envoie un message à llama3.2:3b via Ollama (100% local, 0€)
- Affiche la réponse générée
- Affiche le nombre de tokens consommés en input et output

Pour jouer :
- Change le contenu du message et observe l'évolution des tokens
- Ajoute un "role: system" avant le message user pour donner une personnalité au modèle
- Change le modèle (ex: "llama3.3:70b") et compare la qualité des réponses (optionnel, si votre ordinateur le permet)
- Envoie plusieurs messages à la suite pour simuler une conversation
"""

import ollama

response = ollama.chat(
    model="llama3.2:3b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Explique ce qu'est un embedding en 5 phrases. En 3 phrases supplémentaires, par des matrice d'embedding et de unembedding.",
        },
    ],
)

print(response["message"]["content"])
print(
    f"\nTokens utilisés : {response['prompt_eval_count']} input / {response['eval_count']} output"
)
