"""
Exercice 2 — Premiers embeddings

Objectif : comprendre ce qu'est un embedding — la représentation numérique
du sens d'un texte — et voir concrètement à quoi ça ressemble.

Ce que fait ce code :
- Charge un petit modèle d'embedding en local via sentence-transformers (~90MB)
- Transforme 4 phrases en vecteurs de 384 nombres chacune
- Affiche la dimension et les premières valeurs d'un vecteur

Pour jouer :
- Ajoute tes propres phrases et observe les vecteurs générés
- Change de modèle (ex: "all-mpnet-base-v2") pour des embeddings de meilleure qualité (768 dimensions)
- Essaie avec des phrases en anglais vs français et compare
- Affiche les vecteurs complets de deux phrases similaires côte à côte
"""

from sentence_transformers import SentenceTransformer

# Modèle téléchargé automatiquement en local (~90MB)
model = SentenceTransformer("all-MiniLM-L6-v2")

phrases = [
    "Le chat dort sur le canapé",
    "La chatte sommeille sur le sofa",
    "La bourse monte en flèche",
    "Python est un langage de programmation",
]

embeddings = model.encode(phrases)

print(f"Dimension d'un embedding : {embeddings[0].shape}")  # (384,)
print(f"Les 5 premiers nombres du vecteur 1 : {embeddings[0][:5]}")
print("Les vecteurs de 2 phrases similaires :")
print(f"1: {embeddings[0]}")
print(f"2: {embeddings[1]}")

# Sauvegarde les embeddings dans un fichier
import numpy as np

np.save("./phases/phase1/embeddings.npy", embeddings)
