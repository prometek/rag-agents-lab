"""
Exercice 4 — RAG + Tools combinés

Objectif : combiner recherche documentaire, calculatrice et recherche ETF
dans un seul agent capable de répondre à des questions complexes.

Ce que fait ce code :
- Expose 3 tools : RAG docs, recherche ETF, calculatrice
- L'agent orchestre les tools dans l'ordre qu'il juge nécessaire
- Peut combiner contexte documentaire + données ETF + calcul en une seule réponse

Ce que tu observes :
- L'agent enchaîne plusieurs tools pour les questions complexes
- Il sait quand s'arrêter et quand chercher plus d'informations
- C'est le pattern le plus proche d'un vrai assistant financier

Pour jouer :
- Pose des questions qui nécessitent les 3 tools dans l'ordre
- Observe comment l'agent décompose une question complexe en sous-tâches
- Ajoute un tool "news_marche" (simulé) et observe l'intégration
- C'est la base du projet Obsidian — remplace les docs ETF par tes notes
"""

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
import json

MODEL = ChatOllama(model="llama3.1:8b")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ── Base documentaire ────────────────────────────────────────
documents = [
    Document(
        page_content="Le DCA consiste à investir une somme fixe chaque mois pour lisser le prix d'achat moyen."
    ),
    Document(
        page_content="Il est conseillé de garder 3 à 6 mois de dépenses en épargne de précaution avant d'investir."
    ),
    Document(
        page_content="Le PEA est fiscalement avantageux pour les ETF éligibles après 5 ans."
    ),
    Document(
        page_content="La diversification sur un ETF monde réduit le risque spécifique à un pays ou secteur."
    ),
    Document(
        page_content="L'horizon long terme (10+ ans) permet d'absorber la volatilité des marchés."
    ),
    Document(
        page_content="Le rééquilibrage annuel maintient l'allocation cible du portefeuille."
    ),
]

vectorstore = Chroma.from_documents(
    documents, embeddings, persist_directory="./phases/phase6/chroma_combined"
)


# ── Tools ────────────────────────────────────────────────────
@tool
def rechercher_strategie(query: str) -> str:
    """
    Recherche des conseils et stratégies d'investissement dans la base documentaire.
    Utilise pour toute question sur comment investir, DCA, diversification, horizon.
    """
    docs = vectorstore.as_retriever(search_kwargs={"k": 3}).invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])


@tool
def rechercher_etf(nom: str) -> str:
    """
    Recherche les caractéristiques d'un ETF : TER, éligibilité PEA, type, indice.
    Utilise pour toute question sur un ETF spécifique.
    """
    etfs = {
        "cw8": {
            "nom": "Amundi MSCI World",
            "ter": "0.38%",
            "pea": "Oui",
            "type": "Capitalisant",
            "indice": "MSCI World",
        },
        "amundi msci world": {
            "nom": "Amundi MSCI World",
            "ter": "0.38%",
            "pea": "Oui",
            "type": "Capitalisant",
            "indice": "MSCI World",
        },
        "ewld": {
            "nom": "Lyxor MSCI World",
            "ter": "0.12%",
            "pea": "Non",
            "type": "Distribuant",
            "indice": "MSCI World",
        },
        "lyxor msci world": {
            "nom": "Lyxor MSCI World",
            "ter": "0.12%",
            "pea": "Non",
            "type": "Distribuant",
            "indice": "MSCI World",
        },
    }
    result = etfs.get(nom.lower(), {"erreur": f"ETF '{nom}' non trouvé."})
    return json.dumps(result, ensure_ascii=False)


@tool
def calculatrice(expression: str) -> str:
    """
    Effectue des calculs mathématiques et financiers.
    Utilise pour projections, intérêts composés, conversions.
    Exemples : '500 * 12 * 20' ou 'sum([500 * 1.07**i for i in range(20)])'
    """
    try:
        result = eval(expression)
        if isinstance(result, float):
            return f"Résultat : {result:.2f}"
        return f"Résultat : {result}"
    except Exception as e:
        return f"Erreur : {e}"


# ── Agent ────────────────────────────────────────────────────
memory = MemorySaver()
agent = create_react_agent(
    MODEL,
    tools=[rechercher_strategie, rechercher_etf, calculatrice],
    checkpointer=memory,
)


# ── Test ─────────────────────────────────────────────────────
def run(question: str, thread_id: str = "default"):
    print(f"\nQ: {question}")
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]}, config=config
    )
    print(f"R: {result['messages'][-1].content}")
    print("=" * 60)


questions = [
    "Je veux investir 300€ par mois dans un ETF éligible PEA. Lequel choisir et combien j'aurai après 20 ans avec 7% de rendement ?",
    "Quelle est la meilleure stratégie pour débuter l'investissement avec un petit budget ?",
    "Compare le CW8 et le Lyxor MSCI World, et calcule la différence de frais sur 10 000€ investis sur 10 ans.",
]

for q in questions:
    run(q, thread_id=q[:20])
