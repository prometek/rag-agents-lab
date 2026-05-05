"""
Exercice 4 — Corrective RAG (CRAG)

Objectif : implémenter un pipeline qui évalue la qualité des chunks
récupérés et se corrige automatiquement si ils ne sont pas pertinents.

Ce que fait ce code :
- Récupère les chunks initiaux
- Évalue leur pertinence par rapport à la question (score de confiance)
- Si le score est trop bas : reformule la question et relance la recherche
- Génère la réponse finale avec les meilleurs chunks

Les 3 cas possibles :
- Score élevé (> 0.7)  → les chunks sont bons, on génère directement
- Score moyen (0.3-0.7) → on mélange chunks originaux et reformulés
- Score bas (< 0.3)    → on reformule complètement la question

Pour jouer :
- Change les seuils (0.3 et 0.7) et observe le comportement
- Pose des questions très éloignées du corpus et vois comment le pipeline réagit
- Ajoute un log pour voir combien de fois la correction est déclenchée
- Compare la qualité des réponses avec et sans correction sur les mêmes questions
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
]

chunk_embeddings = EMBED_MODEL.encode(chunks)


# ── Retrieval de base ────────────────────────────────────────
def retrieve(question, k=5):
    q_embedding = EMBED_MODEL.encode([question])[0]
    scores = [
        np.dot(q_embedding, ce) / (np.linalg.norm(q_embedding) * np.linalg.norm(ce))
        for ce in chunk_embeddings
    ]
    top_k = sorted(zip(scores, chunks), reverse=True)[:k]
    return top_k  # retourne (score, chunk)


# ── Évaluation de la pertinence ──────────────────────────────
def evaluate_relevance(question, candidates):
    """Utilise le CrossEncoder pour scorer la pertinence des chunks."""
    pairs = [[question, chunk] for _, chunk in candidates]
    scores = RERANKER.predict(pairs)
    return float(np.mean(scores))


# ── Reformulation de question ────────────────────────────────
def reformulate_question(question):
    """Demande au LLM de reformuler la question pour améliorer le retrieval."""
    prompt = f"""La question suivante n'a pas trouvé de résultats pertinents dans la base de connaissance.
Reformule-la différemment en utilisant d'autres mots, tout en gardant le même sens.
Retourne UNIQUEMENT la question reformulée, sans explication.

Question originale : {question}
Question reformulée :"""

    response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"].strip()


# ── Pipeline Corrective RAG ──────────────────────────────────
def corrective_rag(question, high_threshold=0.7, low_threshold=0.3):
    print(f"\nQ: {question}")

    # Étape 1 : retrieval initial
    candidates = retrieve(question, k=5)
    relevance_score = evaluate_relevance(question, candidates)
    print(f"Score de pertinence initial : {relevance_score:.3f}")

    if relevance_score >= high_threshold:
        # Chunks suffisamment pertinents
        print("→ Chunks pertinents, pas de correction nécessaire")
        best_chunks = [chunk for _, chunk in candidates[:3]]

    elif relevance_score <= low_threshold:
        # Chunks très peu pertinents → reformulation complète
        print("→ Score trop bas, reformulation de la question...")
        new_question = reformulate_question(question)
        print(f"→ Nouvelle question : {new_question}")
        new_candidates = retrieve(new_question, k=5)
        best_chunks = [chunk for _, chunk in new_candidates[:3]]

    else:
        # Score moyen → on mélange les deux
        print("→ Score moyen, enrichissement avec question reformulée...")
        new_question = reformulate_question(question)
        new_candidates = retrieve(new_question, k=3)
        all_candidates = candidates + new_candidates
        # Dédoublonnage
        seen = set()
        unique = []
        for score, chunk in all_candidates:
            if chunk not in seen:
                seen.add(chunk)
                unique.append((score, chunk))
        best_chunks = [chunk for _, chunk in unique[:3]]

    # Étape 2 : génération
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
    "Quelle est la fiscalité du PEA après 5 ans ?",  # bonne question, chunks pertinents
    "Comment fonctionne l'investissement progressif ?",  # question floue, besoin de correction
    "Quel est le meilleur restaurant de Paris ?",  # hors sujet total
]

for q in questions:
    response = corrective_rag(q)
    print(f"Réponse : {response}\n")
    print("-" * 60)
