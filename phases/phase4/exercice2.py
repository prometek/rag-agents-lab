"""
Exercice 2 — Reranking avec CrossEncoder

Objectif : améliorer le classement des chunks récupérés en utilisant
un modèle de reranking plus précis que la similarité cosine.

Ce que fait ce code :
- Récupère les 10 premiers chunks avec la recherche vectorielle (large filet)
- Passe ces chunks dans un CrossEncoder qui les reclasse par pertinence réelle
- Compare le classement avant et après reranking
- Génère la réponse finale avec les 3 meilleurs chunks rerankés

Pourquoi le reranking est plus précis :
- La similarité cosine compare question et chunk indépendamment
- Le CrossEncoder analyse la question ET le chunk ensemble, comme un humain le ferait
- Plus lent mais beaucoup plus précis — c'est pour ça qu'on l'applique
  sur un petit nombre de chunks déjà présélectionnés

Pour jouer :
- Change le nombre de chunks initial (retrieval_k) et final (rerank_k)
- Observe les différences de classement avant/après reranking
- Teste avec des questions ambiguës où le reranking fait le plus de différence
- Compare la réponse finale avec et sans reranking

Dépendances :
- uv add sentence-transformers
  (cross-encoder/ms-marco-MiniLM-L-6-v2 est téléchargé automatiquement ~80MB)
"""

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import ollama

MODEL = "llama3.2:3b"
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

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

chunk_embeddings = EMBED_MODEL.encode(chunks)


# ── Retrieval initial ────────────────────────────────────────
def retrieve(question, k=10):
    q_embedding = EMBED_MODEL.encode([question])[0]
    scores = [
        np.dot(q_embedding, ce) / (np.linalg.norm(q_embedding) * np.linalg.norm(ce))
        for ce in chunk_embeddings
    ]
    top_k = sorted(zip(scores, chunks), reverse=True)[:k]
    return [chunk for _, chunk in top_k]


# ── Reranking ────────────────────────────────────────────────
def rerank(question, candidates, top_k=3):
    pairs = [[question, chunk] for chunk in candidates]
    scores = RERANKER.predict(pairs)
    ranked = sorted(zip(scores, candidates), reverse=True)
    return [chunk for _, chunk in ranked[:top_k]]


# ── Pipeline complet ─────────────────────────────────────────
def rag_with_reranking(question, retrieval_k=10, rerank_k=3):
    # Étape 1 : large retrieval
    candidates = retrieve(question, k=retrieval_k)

    print(f"--- Top 3 AVANT reranking ---")
    for i, chunk in enumerate(candidates[:3]):
        print(f"  {i+1}. {chunk[:70]}...")

    # Étape 2 : reranking
    best_chunks = rerank(question, candidates, top_k=rerank_k)

    print(f"\n--- Top 3 APRÈS reranking ---")
    for i, chunk in enumerate(best_chunks):
        print(f"  {i+1}. {chunk[:70]}...")

    # Étape 3 : génération
    context = "\n\n".join(best_chunks)
    prompt = f"""Voici des extraits de documents :

{context}

En te basant UNIQUEMENT sur ces extraits, réponds à la question :
{question}

Si la réponse n'est pas dans les extraits, dis-le clairement.
"""
    response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]


# ── Test ─────────────────────────────────────────────────────
questions = [
    "Quelle est la fiscalité des ETF sur un PEA ?",
    "Comment investir régulièrement sans risque ?",
]

for q in questions:
    print(f"\n{'='*60}")
    print(f"Q: {q}\n")
    response = rag_with_reranking(q)
    print(f"\nRéponse : {response}")
