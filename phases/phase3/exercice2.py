"""
Exercice 2 — RAG avec ChromaDB

Objectif : remplacer la similarité cosine manuelle par un vrai vector store
et comprendre ce que ChromaDB apporte concrètement.

Ce que fait ce code :
- Reprend le même document et le même chunking que l'exercice 1
- Indexe les chunks dans ChromaDB (stockage local dans ./chroma_db)
- Interroge ChromaDB pour retrouver les chunks pertinents
- Génère la réponse via Ollama

Ce que ChromaDB apporte vs exercice 1 :
- Persistance : les embeddings sont sauvegardés sur disque, pas recalculés à chaque fois
- Scalabilité : fonctionne sur des milliers de documents sans ralentissement
- Metadata : tu peux filtrer par source, date, auteur, etc.

Pour jouer :
- Ajoute des documents supplémentaires à la collection
- Utilise les metadata pour filtrer par source (ex: filtrer uniquement les docs PDF)
- Change n_results et observe l'impact sur la qualité de la réponse
- Inspecte le dossier ./chroma_db pour voir ce qui est stocké sur disque
"""

import ollama
import chromadb
from sentence_transformers import SentenceTransformer

MODEL = "llama3.2:3b"
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# ── Setup ChromaDB ───────────────────────────────────────────
client = chromadb.PersistentClient(path="./phases/phase3/chroma_db")
collection = client.get_or_create_collection("rag_phase3")

# ── Document et chunking (identique exercice 1) ──────────────
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


def chunk_text(text, chunk_size=200, chunk_overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks


chunks = chunk_text(document)

# ── Indexation dans ChromaDB ─────────────────────────────────
embeddings = EMBED_MODEL.encode(chunks).tolist()

collection.upsert(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    documents=chunks,
    embeddings=embeddings,
    metadatas=[{"source": "etf_doc", "chunk_index": i} for i in range(len(chunks))],
)

print(f"{collection.count()} chunks indexés dans ChromaDB\n")


# ── Retrieval + Generation ───────────────────────────────────
def rag(question, n_results=3):
    question_embedding = EMBED_MODEL.encode([question]).tolist()

    results = collection.query(query_embeddings=question_embedding, n_results=n_results)

    context = "\n\n".join(results["documents"][0])

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
]

for q in questions:
    print(f"Q: {q}")
    print(f"R: {rag(q)}\n")
