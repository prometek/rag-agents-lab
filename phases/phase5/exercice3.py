"""
Exercice 3 — Agent avec mémoire + Ollama

Objectif : ajouter un état persistant entre les tours de conversation
pour que l'agent se souvienne du contexte.

Ce que fait ce code :
- Reprend l'agent LangGraph de l'exercice 2
- Ajoute un MemorySaver pour persister l'état entre les appels
- Lance une vraie conversation multi-tours avec contexte

Ce que tu observes :
- L'agent se souvient des questions précédentes
- Il peut faire référence à des informations mentionnées plus tôt
- Chaque thread_id est une conversation indépendante

Pour jouer :
- Lance deux conversations avec des thread_id différents
  et vérifie que les mémoires sont bien isolées
- Pose une question qui fait référence à une réponse précédente
  ("et si je doublais ce montant ?")
- Inspecte le state avec agent.get_state(config) pour voir ce qui est stocké
- Compare la cohérence des réponses avec llama3.2:3b vs llama3.1:8b
"""

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
import json

MODEL = ChatOllama(model="llama3.2:3b")


# ── Tools ────────────────────────────────────────────────────
@tool
def calculatrice(expression: str) -> str:
    """Effectue des calculs mathématiques. Utilise pour tout calcul numérique."""
    try:
        return f"Résultat : {eval(expression):.2f}"
    except Exception as e:
        return f"Erreur : {e}"


@tool
def recherche_etf(nom: str) -> str:
    """Recherche les informations d'un ETF : TER, encours, indice, éligibilité PEA."""
    etfs = {
        "cw8": {
            "nom": "Amundi MSCI World",
            "ter": "0.38%",
            "pea": "Oui",
            "type": "Capitalisant",
        },
        "amundi msci world": {
            "nom": "Amundi MSCI World",
            "ter": "0.38%",
            "pea": "Oui",
            "type": "Capitalisant",
        },
        "lyxor msci world": {
            "nom": "Lyxor MSCI World",
            "ter": "0.12%",
            "pea": "Non",
            "type": "Distribuant",
        },
    }
    key = nom.lower()
    if key in etfs:
        return json.dumps(etfs[key], ensure_ascii=False)
    return f"ETF '{nom}' non trouvé."


# ── Agent avec mémoire ───────────────────────────────────────
memory = MemorySaver()
tools = [calculatrice, recherche_etf]
agent = create_react_agent(MODEL, tools, checkpointer=memory)


# ── Boucle de conversation ───────────────────────────────────
def chat(question: str, thread_id: str = "default"):
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]}, config=config
    )
    return result["messages"][-1].content


# ── Test conversation multi-tours ────────────────────────────
print("=== Conversation avec mémoire ===\n")

thread = "conversation_1"

questions = [
    "Donne moi les infos sur le CW8.",
    "Si j'investis 300€ par mois dedans avec 7% de rendement, combien j'ai après 15 ans ?",
    "Et si je doublais ma mise mensuelle ?",
]

for q in questions:
    print(f"Toi   : {q}")
    response = chat(q, thread_id=thread)
    print(f"Agent : {response}\n")

# ── Vérifier l'isolation des threads ─────────────────────────
print("\n=== Nouvelle conversation (thread différent) ===\n")
response = chat("De quel ETF on parlait ?", thread_id="conversation_2")
print(f"Agent : {response}")
