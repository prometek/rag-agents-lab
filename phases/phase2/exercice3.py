# ─────────────────────────────────────────
# Exercice 3 — Chain-of-Thought
# ─────────────────────────────────────────
"""
Objectif : comprendre comment forcer le modèle à raisonner étape par étape
réduit les erreurs sur des tâches complexes.

Ce que fait ce code :
- Pose un problème de calcul sans instruction de raisonnement
- Pose le même problème en demandant "étape par étape"
- Compare les deux réponses

Pour jouer :
- Teste avec un problème plus complexe (plusieurs conditions, pourcentages)
- Essaie "pense à voix haute avant de répondre" vs "étape par étape" — observe la différence
- Teste sur une question d'analyse (pas juste du calcul) pour voir si CoT aide aussi
"""

import ollama

MODEL = "llama3.2:3b"

probleme = "J'achète 3 ETF à 120€ chacun. Chacun monte de 8%. Quelle est ma plus-value totale ?"

sans_cot = probleme
avec_cot = f"{probleme} Raisonne étape par étape."

r1 = ollama.chat(model=MODEL, messages=[{"role": "user", "content": sans_cot}])
r2 = ollama.chat(model=MODEL, messages=[{"role": "user", "content": avec_cot}])

print("=== Sans CoT ===")
print(r1["message"]["content"])
print("\n=== Avec CoT ===")
print(r2["message"]["content"])
