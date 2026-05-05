"""
Exercice 1 — RAG as a Tool

Objectif : transformer le pipeline RAG en tool LangGraph pour que
l'agent décide lui-même quand chercher dans les documents.

Ce que fait ce code :
- Crée un index vectoriel avec les documents ETF
- Expose le retriever comme un tool LangGraph (@tool)
- L'agent décide quand appeler ce tool vs répondre directement

Ce que tu observes :
- Sur une question documentaire → l'agent appelle le tool RAG
- Sur une question générale → l'agent répond directement sans chercher
- C'est le comportement intelligent qu'on veut en production

Pour jouer :
- Pose des questions générales (météo, définition) et observe que l'agent
  ne cherche pas dans les docs
- Pose des questions très spécifiques et observe combien de fois il cherche
- Ajoute un deuxième tool (calculatrice) et pose des questions mixtes
- Change k dans le retriever et observe l'impact sur la qualité
"""

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

MODEL = ChatOllama(model="llama3.1:8b")

# ── Base documentaire ────────────────────────────────────────
documents = [
    Document(
        page_content="Les ETF répliquent passivement un indice boursier comme le CAC 40 ou le S&P 500."
    ),
    Document(
        page_content="Le TER des ETF est généralement inférieur à 0.5% par an. Les fonds actifs facturent 1.5% à 2%."
    ),
    Document(
        page_content="Les ETF distribuants versent des dividendes, les ETF capitalisants les réinvestissent automatiquement."
    ),
    Document(
        page_content="Le PEA permet d'investir dans des ETF éligibles avec une exonération d'IR après 5 ans."
    ),
    Document(
        page_content="La stratégie DCA consiste à investir une somme fixe à intervalle régulier pour lisser le prix d'achat."
    ),
    Document(
        page_content="Le MSCI World couvre plus de 1500 entreprises dans 23 pays développés."
    ),
    Document(
        page_content="Le CTO est soumis au PFU de 30% sur les plus-values. Le PEA est plus avantageux fiscalement."
    ),
    Document(
        page_content="L'Amundi MSCI World (CW8) a un TER de 0.38% et est éligible au PEA."
    ),
]

# ── Création du vector store ─────────────────────────────────
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    documents, embeddings, persist_directory="./phases/phase6/chroma_agentic"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# ── RAG Tool ─────────────────────────────────────────────────
@tool
def rechercher_dans_docs(query: str) -> str:
    """
    Recherche des informations dans la base documentaire sur les ETF,
    l'investissement, la fiscalité PEA/CTO, et les stratégies financières.
    Utilise cet outil pour toute question sur ces sujets.
    """
    docs = retriever.invoke(query)
    if not docs:
        return "Aucun document pertinent trouvé."
    return "\n\n".join([doc.page_content for doc in docs])


# ── Agent ────────────────────────────────────────────────────
memory = MemorySaver()
agent = create_react_agent(MODEL, tools=[rechercher_dans_docs], checkpointer=memory)


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
    "C'est quoi la différence entre un ETF distribuant et capitalisant ?",
    "Quelle est la capitale de la France ?",  # hors docs → réponse directe
    "Le CW8 est-il éligible PEA ?",
]

for q in questions:
    run(q)
