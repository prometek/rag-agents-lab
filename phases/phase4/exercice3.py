"""
Exercice 3 — Évaluation avec RAGAS

Objectif : mesurer objectivement la qualité de ton pipeline RAG
avec 4 métriques standards.

Ce que fait ce code :
- Prépare un dataset de questions/réponses de référence
- Fait tourner le pipeline RAG sur ces questions
- Évalue les résultats avec RAGAS
- Affiche les 4 métriques clés

Les 4 métriques RAGAS :
- Context Precision   : les chunks récupérés sont-ils pertinents ?
- Context Recall      : tous les chunks nécessaires ont-ils été récupérés ?
- Answer Relevancy    : la réponse répond-elle à la question ?
- Faithfulness        : la réponse est-elle fidèle aux chunks (pas d'hallucination) ?

Score de 0 à 1 pour chaque métrique — viser > 0.7 en production.

Pour jouer :
- Change chunk_size et observe l'impact sur les métriques
- Compare les scores avec et sans reranking (exercice 2)
- Ajoute des questions pour lesquelles la réponse n'est pas dans les docs
  et observe comment Faithfulness réagit
- Essaie de dégrader volontairement le pipeline et vois quelle métrique chute en premier

Dépendances :
- uv add ragas datasets
- Une clé API OpenAI est requise par RAGAS pour l'évaluation
  (RAGAS utilise un LLM pour évaluer — ~$0.10 pour ce test)
  Alternative : voir la doc RAGAS pour utiliser un modèle local
"""

import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (  # noqa: old path, but required for LangchainLLMWrapper support
    ContextPrecision,
    ContextRecall,
    AnswerRelevancy,
    Faithfulness,
)
from ragas.metrics.base import Metric
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import numpy as np
import ollama

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
MODEL = "llama3.1:8b"

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


# ── Pipeline RAG ─────────────────────────────────────────────
def retrieve(question, k=3):
    q_embedding = EMBED_MODEL.encode([question])[0]
    scores = [
        np.dot(q_embedding, ce) / (np.linalg.norm(q_embedding) * np.linalg.norm(ce))
        for ce in chunk_embeddings
    ]
    top_k = sorted(zip(scores, chunks), reverse=True)[:k]
    return [chunk for _, chunk in top_k]


def generate(question, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""Voici des extraits de documents :

{context}

En te basant UNIQUEMENT sur ces extraits, réponds à la question :
{question}

Si la réponse n'est pas dans les extraits, dis-le clairement.
"""
    response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]


# ── Dataset d'évaluation ─────────────────────────────────────
# questions  : les questions à évaluer
# ground_truth : les réponses de référence attendues
eval_questions = [
    "Quelle est la différence entre un ETF distribuant et capitalisant ?",
    "Quel est l'avantage fiscal du PEA pour les ETF ?",
    "Qu'est-ce que le TER ?",
    "Quelle est la stratégie DCA ?",
]

ground_truths = [
    "Les ETF distribuants versent des dividendes aux investisseurs, tandis que les ETF capitalisants réinvestissent automatiquement les dividendes.",
    "Le PEA offre une fiscalité avantageuse après 5 ans de détention pour les ETF éligibles.",
    "Le TER (Total Expense Ratio) représente les frais de gestion annuels d'un ETF, généralement inférieurs à 0.5% par an.",
    "La stratégie DCA (Dollar Cost Averaging) consiste à investir une somme fixe à intervalle régulier.",
]

# ── Générer les réponses et collecter les contextes ──────────
answers = []
contexts = []

for question in eval_questions:
    retrieved = retrieve(question, k=3)
    answer = generate(question, retrieved)
    answers.append(answer)
    contexts.append(retrieved)

# ── Évaluation RAGAS ─────────────────────────────────────────
dataset = Dataset.from_dict(
    {
        "question": eval_questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }
)

ragas_llm = LangchainLLMWrapper(ChatOllama(model=MODEL))
ragas_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)

context_precision = ContextPrecision(llm=ragas_llm)
context_recall = ContextRecall(llm=ragas_llm)
answer_relevancy = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
faithfulness = Faithfulness(llm=ragas_llm)

run_config = RunConfig(
    timeout=300,  # 5 min par job — modèle local lent
    max_workers=1,  # pas de parallélisme pour éviter les timeouts en rafale
    max_retries=3,
)

results = evaluate(
    dataset,
    metrics=[context_precision, context_recall, answer_relevancy, faithfulness],
    run_config=run_config,
)


def mean(scores):
    valid = [s for s in scores if s == s]  # exclut les NaN
    return sum(valid) / len(valid) if valid else float("nan")


print("\n=== Résultats RAGAS ===")
print(f"Context Precision  : {mean(results['context_precision']):.3f}")
print(f"Context Recall     : {mean(results['context_recall']):.3f}")
print(f"Answer Relevancy   : {mean(results['answer_relevancy']):.3f}")
print(f"Faithfulness       : {mean(results['faithfulness']):.3f}")

metric_names = ["context_precision", "context_recall", "answer_relevancy", "faithfulness"]
all_scores = [mean(results[m]) for m in metric_names]
valid_scores = [s for s in all_scores if s == s]
if valid_scores:
    print(f"\nScore global       : {sum(valid_scores) / len(valid_scores):.3f}")
