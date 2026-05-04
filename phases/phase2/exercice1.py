# ─────────────────────────────────────────
# Exercice 1 — Zero-shot vs Few-shot
# ─────────────────────────────────────────
"""
Objectif : comprendre l'impact des exemples sur la qualité de la réponse.
On envoie la même tâche deux fois — sans exemple, puis avec — et on compare.

Ce que fait ce code :
- Envoie une tâche de classification sans exemple (zero-shot)
- Envoie la même tâche avec 3 exemples (few-shot)
- Affiche les deux réponses côte à côte

Pour jouer :
- Change le texte à classifier
- Ajoute ou retire des exemples et observe l'évolution
- Teste avec une tâche plus ambiguë pour voir quand le few-shot fait vraiment la différence
"""

import ollama

MODEL = "llama3.2:3b"

texte = "Livraison rapide mais emballage abîmé, dommage."

zero_shot = f"Classe ce texte comme positif, négatif ou neutre : '{texte}'"

few_shot = f"""
Classe ce texte comme positif, négatif ou neutre.

Exemples :
- "Super produit, je recommande" → positif
- "Complètement nul, à éviter" → négatif
- "Correct sans plus" → neutre

Texte : '{texte}'
→"""

r1 = ollama.chat(model=MODEL, messages=[{"role": "user", "content": zero_shot}])
r2 = ollama.chat(model=MODEL, messages=[{"role": "user", "content": few_shot}])

print("=== Zero-shot ===")
print(r1["message"]["content"])
print("\n=== Few-shot ===")
print(r2["message"]["content"])
