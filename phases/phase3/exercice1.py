"""
Exercice 1 — RAG from scratch (sans framework)

Objectif : construire le pipeline RAG complet à la main pour comprendre
chaque brique sans abstraction. C'est l'exercice le plus important de la phase.

Pipeline :
Documents → Chunking manuel → Embeddings → Similarité cosine → Prompt → Réponse

Ce que fait ce code :
- Découpe un texte en chunks avec overlap
- Génère les embeddings de chaque chunk
- Reçoit une question, calcule son embedding
- Trouve les chunks les plus proches par similarité cosine
- Construit un prompt avec ces chunks et interroge Ollama

Pour jouer :
- Change chunk_size et chunk_overlap et observe l'impact sur la précision
- Augmente k (nombre de chunks récupérés) et vois si la réponse s'améliore
- Pose une question dont la réponse n'est pas dans le texte — observe comment
  le modèle gère l'absence d'information
- Remplace le texte par un de tes propres documents
"""

import ollama
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL = "llama3.2:3b"
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# ── Document source ──────────────────────────────────────────
document = """
Les ETF (Exchange Traded Funds) sont des fonds d'investissement cotés en bourse.
Ils permettent d'investir dans un panier de valeurs en une seule transaction.
Les ETF répliquent généralement un indice comme le CAC 40 ou le S&P 500.
Leurs frais de gestion sont très faibles, souvent inférieurs à 0.5% par an.
Contrairement aux fonds actifs, les ETF ne cherchent pas à battre le marché.
Un ETF peut être acheté et vendu tout au long de la journée comme une action.
La diversification offerte par les ETF réduit le risque global du portefeuille.
Les ETF distribuants versent des dividendes, les ETF capitalisants les réinvestissent.
Le PEA permet d'investir dans des ETF européens avec une fiscalité avantageuse.
"""


# ── Chunking manuel ──────────────────────────────────────────
def chunk_text(text, chunk_size=200, chunk_overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks


chunks = chunk_text(document)
print(f"{len(chunks)} chunks générés\n")

# ── Embeddings ───────────────────────────────────────────────
chunk_embeddings = EMBED_MODEL.encode(chunks)


# ── Similarité cosine ────────────────────────────────────────
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def retrieve(question, k=3):
    question_embedding = EMBED_MODEL.encode([question])[0]
    scores = [cosine_similarity(question_embedding, ce) for ce in chunk_embeddings]
    top_k = sorted(zip(scores, chunks), reverse=True)[:k]
    return [chunk for _, chunk in top_k]


# ── Generation ───────────────────────────────────────────────
def rag(question, k=3):
    context_chunks = retrieve(question, k)
    context = "\n\n".join(context_chunks)

    prompt = f"""Voici des extraits de documents :

{context}

En te basant UNIQUEMENT sur ces extraits, réponds à la question suivante :
{question}

Si la réponse n'est pas dans les extraits, dis-le clairement.
"""
    response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]


# ── Test ─────────────────────────────────────────────────────
questions = [
    "Quelle est la différence entre un ETF distribuant et capitalisant ?",
    "Comment fonctionne la fiscalité des ETF en France ?",
    "Quel est le taux de rendement moyen d'un ETF ?",  # pas dans le document
]

for q in questions:
    print(f"Q: {q}")
    print(f"R: {rag(q)}\n")
