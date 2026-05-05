"""
Exercice 4 — Agent supervisor + Ollama

Objectif : créer un agent supervisor qui délègue les tâches
à des agents spécialisés selon la nature de la question.

Ce que fait ce code :
- Agent Finance : répond aux questions sur les ETF et fiscalité
- Agent Calcul  : effectue les calculs financiers
- Supervisor    : analyse la question et route vers le bon agent

Note : llama3.1:8b est utilisé pour le supervisor car le routing
nécessite plus de capacité de raisonnement que llama3.2:3b.

Pour jouer :
- Ajoute un Agent Fiscal spécialisé sur la fiscalité PEA/CTO
- Teste avec des questions qui nécessitent les deux agents
- Observe comment le supervisor décompose une question complexe
- Remplace llama3.1:8b par llama3.2:3b sur le supervisor et compare la qualité du routing
"""

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from typing import TypedDict, Literal
import json

# llama3.1:8b pour le supervisor (meilleur raisonnement)
# llama3.2:3b pour les agents spécialisés (plus rapide)
SUPERVISOR_MODEL = ChatOllama(model="llama3.1:8b")
AGENT_MODEL = ChatOllama(model="llama3.2:3b")


# ── Tools par domaine ────────────────────────────────────────
@tool
def recherche_etf(nom: str) -> str:
    """Recherche les informations d'un ETF."""
    etfs = {
        "cw8": {"nom": "Amundi MSCI World", "ter": "0.38%", "pea": "Oui"},
        "amundi msci world": {"nom": "Amundi MSCI World", "ter": "0.38%", "pea": "Oui"},
        "lyxor msci world": {"nom": "Lyxor MSCI World", "ter": "0.12%", "pea": "Non"},
    }
    return json.dumps(
        etfs.get(nom.lower(), {"erreur": "ETF non trouvé"}), ensure_ascii=False
    )


@tool
def info_fiscalite(enveloppe: str) -> str:
    """Retourne les informations fiscales d'une enveloppe (PEA, CTO, assurance-vie)."""
    fiscalite = {
        "pea": "Exonération IR après 5 ans. Prélèvements sociaux 17.2% uniquement.",
        "cto": "PFU 30% (flat tax) sur les plus-values et dividendes.",
        "assurance-vie": "Fiscalité avantageuse après 8 ans. Abattement annuel de 4600€.",
    }
    return fiscalite.get(enveloppe.lower(), "Enveloppe non reconnue.")


@tool
def calculatrice(expression: str) -> str:
    """Effectue des calculs mathématiques."""
    try:
        return f"Résultat : {eval(expression):.2f}"
    except Exception as e:
        return f"Erreur : {e}"


# ── Agents spécialisés ───────────────────────────────────────
finance_agent = create_react_agent(
    AGENT_MODEL,
    tools=[recherche_etf, info_fiscalite],
    prompt=SystemMessage(
        content="Tu es un expert en ETF et fiscalité. Réponds uniquement sur ces sujets."
    ),
)

calcul_agent = create_react_agent(
    AGENT_MODEL,
    tools=[calculatrice],
    prompt=SystemMessage(
        content="Tu es un expert en calculs financiers. Effectue les calculs demandés avec précision."
    ),
)


# ── State du graphe ──────────────────────────────────────────
class SupervisorState(TypedDict):
    question: str
    agent_choisi: str
    reponse_agent: str


# ── Node supervisor ──────────────────────────────────────────
def supervisor_node(state: SupervisorState) -> SupervisorState:
    prompt = f"""Tu es un supervisor qui route les questions vers le bon agent.

Agents disponibles :
- "finance"  : questions sur les ETF, indices, fiscalité PEA/CTO
- "calcul"   : calculs mathématiques et financiers
- "les_deux" : question qui nécessite les deux agents

Question : {state['question']}

Réponds UNIQUEMENT avec un seul mot : finance, calcul, ou les_deux"""

    response = SUPERVISOR_MODEL.invoke([HumanMessage(content=prompt)])
    agent_choisi = response.content.strip().lower()

    if agent_choisi not in ["finance", "calcul", "les_deux"]:
        agent_choisi = "finance"

    print(f"→ Supervisor route vers : {agent_choisi}")
    return {**state, "agent_choisi": agent_choisi}


# ── Nodes agents ─────────────────────────────────────────────
def finance_node(state: SupervisorState) -> SupervisorState:
    result = finance_agent.invoke(
        {"messages": [{"role": "user", "content": state["question"]}]}
    )
    return {**state, "reponse_agent": result["messages"][-1].content}


def calcul_node(state: SupervisorState) -> SupervisorState:
    result = calcul_agent.invoke(
        {"messages": [{"role": "user", "content": state["question"]}]}
    )
    return {**state, "reponse_agent": result["messages"][-1].content}


def les_deux_node(state: SupervisorState) -> SupervisorState:
    finance_result = finance_agent.invoke(
        {"messages": [{"role": "user", "content": state["question"]}]}
    )
    finance_response = finance_result["messages"][-1].content

    calcul_result = calcul_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"{state['question']}\nContexte : {finance_response}",
                }
            ]
        }
    )

    combined = f"Infos financières :\n{finance_response}\n\nCalculs :\n{calcul_result['messages'][-1].content}"
    return {**state, "reponse_agent": combined}


# ── Routing ──────────────────────────────────────────────────
def route(state: SupervisorState) -> Literal["finance", "calcul", "les_deux"]:
    return state["agent_choisi"]


# ── Construction du graphe ───────────────────────────────────
graph = StateGraph(SupervisorState)

graph.add_node("supervisor", supervisor_node)
graph.add_node("finance", finance_node)
graph.add_node("calcul", calcul_node)
graph.add_node("les_deux", les_deux_node)

graph.set_entry_point("supervisor")
graph.add_conditional_edges("supervisor", route)
graph.add_edge("finance", END)
graph.add_edge("calcul", END)
graph.add_edge("les_deux", END)

app = graph.compile()


# ── Test ─────────────────────────────────────────────────────
def run(question: str):
    print(f"\nQ: {question}")
    result = app.invoke(
        {
            "question": question,
            "agent_choisi": "",
            "reponse_agent": "",
        }
    )
    print(f"Réponse : {result['reponse_agent']}")
    print("=" * 60)


questions = [
    "Quels sont les avantages fiscaux du PEA ?",
    "Si j'investis 400€ par mois avec 7% de rendement pendant 25 ans, combien j'aurai ?",
    "Le CW8 est-il éligible PEA ? Et si j'y investis 500€/mois pendant 20 ans avec 8% ?",
]

for q in questions:
    run(q)
