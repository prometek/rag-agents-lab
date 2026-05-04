"""
Exercice 3 — Similarité cosine

Objectif : mesurer à quel point deux phrases sont sémantiquement proches,
et comprendre le mécanisme central du RAG — trouver les chunks les plus
pertinents par rapport à une question.

Ce que fait ce code :
- Implémente la similarité cosine à la main (sans librairie)
- Compare les embeddings de l'exercice 2 deux à deux
- Affiche un score entre 0 (aucun rapport) et 1 (identique)

Pour jouer :
- Ajoute une phrase ambiguë et vois avec laquelle elle est la plus proche
- Teste avec une question ("Où dort le chat ?") et vois si elle se rapproche
  de la bonne phrase — c'est exactement ce que fait un moteur RAG
- Remplace la formule cosine par une distance euclidienne et compare les résultats
- Trie les phrases par similarité décroissante par rapport à une phrase de référence
"""

import numpy as np

# Charge les embeddings de l'exercice 2
embeddings = np.load("./phases/phase1/embeddings.npy")


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)


# Reprend les embeddings de l'exercice 2
e0 = embeddings[0]  # "Le chat dort sur le canapé"
e1 = embeddings[1]  # "La chatte sommeille sur le sofa"
e2 = embeddings[2]  # "La bourse monte en flèche"

print(f"Chat vs Chatte  : {cosine_similarity(e0, e1):.4f}")  # proche de 1
print(f"Chat vs Bourse  : {cosine_similarity(e0, e2):.4f}")  # proche de 0
print(f"Chatte vs Bourse: {cosine_similarity(e1, e2):.4f}")  # proche de 0
