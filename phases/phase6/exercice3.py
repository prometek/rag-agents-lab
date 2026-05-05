"""
Exercice 3 — Iterative Retrieval

Objectif : l'agent évalue la qualité des chunks récupérés et relance
une recherche reformulée si les résultats sont insuffisants.

Ce que fait ce code :
- Construit un graphe LangGraph custom (pas create_react_agent)
- Node retrieve : cherche les chunks
- Node evaluate : le LLM évalue si les chunks sont suffisants
- Node reformulate : reformule la question si besoin
- Node generate : génère la réponse finale
- Boucle jusqu'à satisfaction ou max_retries atteint

Pour jouer :
- Change max_retries et observe combien d'itérations sont nécessaires
- Pose des questions très spécifiques non présentes dans les docs
  et observe les reformulations successives
- Ajoute un log de toutes les reformulations pour analyser le comportement
- Compare avec le Corrective RAG Phase 4 — quelles différences ?
"""

from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

MODEL = ChatOllama(model="llama3.1:8b")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ── Base documentaire ────────────────────────────────────────
documents = [
    Document(
        page_content="Le CW8 (Amundi MSCI World) a un TER de 0.38% et est éligible PEA."
    ),
    Document(page_content="Le TER des ETF est généralement inférieur à 0.5% par an."),
    Document(
        page_content="Les ETF distribuants versent des dividendes, les capitalisants les réinvestissent."
    ),
    Document(
        page_content="Le PEA offre une exonération d'IR après 5 ans de détention."
    ),
    Document(
        page_content="La stratégie DCA consiste à investir une somme fixe chaque mois."
    ),
    Document(
        page_content="Le MSCI World couvre 1500 entreprises dans 23 pays développés."
    ),
]

vectorstore = Chroma.from_documents(
    documents, embeddings, persist_directory="./phases/phase6/chroma_iterative"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# ── State ────────────────────────────────────────────────────
class RAGState(TypedDict):
    question: str
    query_courante: str
    chunks: List[str]
    qualite_ok: bool
    reponse: str
    retries: int
    max_retries: int


# ── Nodes ────────────────────────────────────────────────────
def retrieve_node(state: RAGState) -> RAGState:
    print(f"→ Recherche : '{state['query_courante']}'")
    docs = retriever.invoke(state["query_courante"])
    chunks = [doc.page_content for doc in docs]
    return {**state, "chunks": chunks}


def evaluate_node(state: RAGState) -> RAGState:
    context = "\n".join(state["chunks"])
    prompt = f"""Tu évalues si ces extraits permettent de répondre à la question.

Question : {state['question']}

Extraits :
{context}

Ces extraits sont-ils suffisants pour répondre ? Réponds UNIQUEMENT par 'oui' ou 'non'."""

    response = MODEL.invoke([HumanMessage(content=prompt)])
    qualite_ok = "oui" in response.content.lower()
    print(f"→ Qualité suffisante : {qualite_ok}")
    return {**state, "qualite_ok": qualite_ok}


def reformulate_node(state: RAGState) -> RAGState:
    prompt = f"""La recherche '{state['query_courante']}' n'a pas trouvé de résultats suffisants.
Reformule cette question avec d'autres mots pour améliorer la recherche.
Retourne UNIQUEMENT la question reformulée."""

    response = MODEL.invoke([HumanMessage(content=prompt)])
    new_query = response.content.strip()
    print(f"→ Reformulation : '{new_query}'")
    return {**state, "query_courante": new_query, "retries": state["retries"] + 1}


def generate_node(state: RAGState) -> RAGState:
    context = "\n\n".join(state["chunks"])
    prompt = f"""Voici des extraits de documents :

{context}

En te basant UNIQUEMENT sur ces extraits, réponds à la question :
{state['question']}

Si la réponse n'est pas dans les extraits, dis-le clairement."""

    response = MODEL.invoke([HumanMessage(content=prompt)])
    return {**state, "reponse": response.content}


# ── Routing ──────────────────────────────────────────────────
def should_continue(state: RAGState) -> str:
    if state["qualite_ok"] or state["retries"] >= state["max_retries"]:
        return "generate"
    return "reformulate"


# ── Graphe ───────────────────────────────────────────────────
graph = StateGraph(RAGState)

graph.add_node("retrieve", retrieve_node)
graph.add_node("evaluate", evaluate_node)
graph.add_node("reformulate", reformulate_node)
graph.add_node("generate", generate_node)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "evaluate")
graph.add_conditional_edges("evaluate", should_continue)
graph.add_edge("reformulate", "retrieve")
graph.add_edge("generate", END)

app = graph.compile()


# ── Test ─────────────────────────────────────────────────────
def run(question: str):
    print(f"\nQ: {question}")
    result = app.invoke(
        {
            "question": question,
            "query_courante": question,
            "chunks": [],
            "qualite_ok": False,
            "reponse": "",
            "retries": 0,
            "max_retries": 3,
        }
    )
    print(f"R: {result['reponse']}")
    print("=" * 60)


questions = [
    "Quel ETF choisir pour un PEA ?",
    "Comment investir progressivement chaque mois ?",
    "Quels sont les frais annuels du CW8 ?",
]

for q in questions:
    run(q)
