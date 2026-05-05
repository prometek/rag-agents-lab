"""
Exercice 1 — Hybrid Search (BM25 + Vector Search)

Objectif : combiner la recherche sémantique (embeddings) et la recherche
par mots-clés (BM25) pour améliorer la précision du retrieval.

Ce que fait ce code :
- Implémente une recherche vectorielle classique
- Implémente une recherche BM25 (mots-clés exacts)
- Combine les deux scores avec un paramètre alpha
- Compare les résultats des trois approches sur les mêmes questions

Pourquoi c'est important :
- Vector search : trouve les idées similaires mais rate les termes exacts
- BM25 : trouve les termes exacts mais ne comprend pas le sens
- Hybrid : le meilleur des deux mondes

Pour jouer :
- Change alpha (0 = BM25 pur, 1 = vector pur, 0.5 = équilibré)
  et observe comment les résultats changent
- Teste avec des questions techniques (acronymes, noms propres)
  où BM25 apporte le plus
- Teste avec des questions sémantiques ("comment réduire le risque ?")
  où le vector search apporte le plus
- Ajoute plus de documents pour voir l'impact sur la précision

Dépendances :
- uv add rank-bm25
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import ollama

MODEL = "llama3.2:3b"
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# ── Documents source ─────────────────────────────────────────
chunks = [
    "Les ETF répliquent passivement un indice boursier comme le CAC 40 ou le S&P 500.",
    "Le TER (Total Expense Ratio) des ETF est généralement inférieur à 0.5% par an.",
    "Les ETF distribuants versent des dividendes, les ETF capitalisants les réinvestissent.",
    "Le PEA permet d'investir dans des ETF éligibles avec une fiscalité avantageuse après 5 ans.",
    "La stratégie DCA consiste à investir une somme fixe à intervalle régulier.",
    "Le MSCI World couvre plus de 1500 entreprises dans 23 pays développés.",
    "Les frais de gestion des fonds actifs sont en moyenne de 1.5% à 2% par an.",
    "Le Compte-Titres Ordinaire permet d'investir dans tous types d'ETF sans restriction.",
    "La réplication physique signifie que le fonds détient réellement les actions de l'indice.",
    "Le prélèvement forfaitaire unique (PFU) de 30% s'applique aux plus-values sur CTO.",
]

# ── Vector Search ────────────────────────────────────────────
chunk_embeddings = EMBED_MODEL.encode(chunks)


def vector_search(question, k=3):
    q_embedding = EMBED_MODEL.encode([question])[0]
    scores = [
        np.dot(q_embedding, ce) / (np.linalg.norm(q_embedding) * np.linalg.norm(ce))
        for ce in chunk_embeddings
    ]
    top_k = sorted(zip(scores, chunks), reverse=True)[:k]
    return top_k


# ── BM25 Search ──────────────────────────────────────────────
tokenized_chunks = [chunk.lower().split() for chunk in chunks]
bm25 = BM25Okapi(tokenized_chunks)


def bm25_search(question, k=3):
    tokenized_question = question.lower().split()
    scores = bm25.get_scores(tokenized_question)
    top_k = sorted(zip(scores, chunks), reverse=True)[:k]
    return top_k


# ── Hybrid Search ────────────────────────────────────────────
def hybrid_search(question, k=3, alpha=0.5):
    """
    alpha = 0   → BM25 pur
    alpha = 1   → Vector pur
    alpha = 0.5 → équilibré
    """
    # Scores vectoriels normalisés
    q_embedding = EMBED_MODEL.encode([question])[0]
    vector_scores = np.array(
        [
            np.dot(q_embedding, ce) / (np.linalg.norm(q_embedding) * np.linalg.norm(ce))
            for ce in chunk_embeddings
        ]
    )
    vector_scores = (vector_scores - vector_scores.min()) / (
        vector_scores.max() - vector_scores.min()
    )

    # Scores BM25 normalisés
    bm25_scores = np.array(bm25.get_scores(question.lower().split()))
    if bm25_scores.max() > 0:
        bm25_scores = bm25_scores / bm25_scores.max()

    # Combinaison
    hybrid_scores = alpha * vector_scores + (1 - alpha) * bm25_scores
    top_k = sorted(zip(hybrid_scores, chunks), reverse=True)[:k]
    return top_k


# ── Comparaison ──────────────────────────────────────────────
questions = [
    "Quel est le TER d'un ETF ?",  # acronyme → BM25 devrait aider
    "Comment réduire le risque de mon portefeuille ?",  # sémantique → vector devrait aider
]

for q in questions:
    print(f"\n{'='*60}")
    print(f"Q: {q}")

    print("\n--- Vector Search ---")
    for score, chunk in vector_search(q):
        print(f"  [{score:.3f}] {chunk[:60]}...")

    print("\n--- BM25 ---")
    for score, chunk in bm25_search(q):
        print(f"  [{score:.3f}] {chunk[:60]}...")

    print("\n--- Hybrid (alpha=0.5) ---")
    for score, chunk in hybrid_search(q, alpha=0.5):
        print(f"  [{score:.3f}] {chunk[:60]}...")
