"""
Exercice 2 — Multi-index Routing

Objectif : créer plusieurs bases vectorielles spécialisées et laisser
l'agent choisir dans laquelle chercher selon la question.

Ce que fait ce code :
- Crée deux index : un sur les ETF, un sur les stratégies d'investissement
- Expose chaque index comme un tool distinct avec une description précise
- L'agent route vers le bon index selon la nature de la question

Ce que tu observes :
- L'agent choisit le bon index selon le type de question
- Les descriptions des tools sont cruciales pour le routing
- Préfigure exactement le projet Obsidian (plusieurs index par dossier PARA)

Pour jouer :
- Ajoute un troisième index (ex: fiscalité)
- Teste avec des questions ambiguës qui pourraient matcher plusieurs index
- Modifie les descriptions des tools et observe si le routing change
- Pose une question qui nécessite les deux index et vois ce que l'agent fait
"""

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langgraph.prebuilt import create_react_agent

MODEL = ChatOllama(model="llama3.1:8b")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ── Index 1 : ETF ────────────────────────────────────────────
docs_etf = [
    Document(
        page_content="Le CW8 (Amundi MSCI World) a un TER de 0.38% et est éligible PEA."
    ),
    Document(
        page_content="Le Lyxor MSCI World a un TER de 0.12% mais n'est pas éligible PEA."
    ),
    Document(
        page_content="Les ETF capitalisants réinvestissent les dividendes automatiquement."
    ),
    Document(
        page_content="Les ETF distribuants versent des dividendes régulièrement aux investisseurs."
    ),
    Document(
        page_content="Le MSCI World couvre 1500 entreprises dans 23 pays développés."
    ),
]

index_etf = Chroma.from_documents(
    docs_etf, embeddings, persist_directory="./phases/phase6/chroma_etf"
)

# ── Index 2 : Stratégies ─────────────────────────────────────
docs_strategie = [
    Document(
        page_content="Le DCA (Dollar Cost Averaging) consiste à investir une somme fixe chaque mois."
    ),
    Document(
        page_content="La diversification réduit le risque en répartissant les investissements sur plusieurs actifs."
    ),
    Document(
        page_content="L'horizon d'investissement long terme (10+ ans) réduit l'impact de la volatilité."
    ),
    Document(
        page_content="Il est conseillé de garder 3 à 6 mois de dépenses en épargne de précaution avant d'investir."
    ),
    Document(
        page_content="Le rééquilibrage annuel permet de maintenir l'allocation cible de son portefeuille."
    ),
]

index_strategie = Chroma.from_documents(
    docs_strategie, embeddings, persist_directory="./phases/phase6/chroma_strategie"
)


# ── Tools spécialisés ────────────────────────────────────────
@tool
def rechercher_etf(query: str) -> str:
    """
    Recherche des informations sur des ETF spécifiques : TER, éligibilité PEA,
    type (capitalisant/distribuant), indices répliqués, encours.
    Utilise pour toute question sur un ETF précis.
    """
    docs = index_etf.as_retriever(search_kwargs={"k": 3}).invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])


@tool
def rechercher_strategie(query: str) -> str:
    """
    Recherche des informations sur les stratégies d'investissement :
    DCA, diversification, horizon de placement, rééquilibrage, épargne de précaution.
    Utilise pour toute question sur comment investir.
    """
    docs = index_strategie.as_retriever(search_kwargs={"k": 3}).invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])


# ── Agent ────────────────────────────────────────────────────
agent = create_react_agent(MODEL, tools=[rechercher_etf, rechercher_strategie])


# ── Test ─────────────────────────────────────────────────────
def run(question: str):
    print(f"\nQ: {question}")
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    print(f"R: {result['messages'][-1].content}")
    print("=" * 60)


questions = [
    "Quelle est la différence entre le CW8 et le Lyxor MSCI World ?",  # → index ETF
    "Comment dois-je commencer à investir avec 200€ par mois ?",  # → index stratégie
    "Est-ce que je devrais acheter du CW8 en DCA ?",  # → les deux
]

for q in questions:
    run(q)
