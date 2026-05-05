"""
Exercice 2 — Agent ReAct avec LangGraph + Ollama

Objectif : reconstruire le même agent qu'en exercice 1 mais avec LangGraph,
et comprendre ce que le framework apporte.

Ce que fait ce code :
- Définit les mêmes tools avec le décorateur @tool
- Crée un graphe LangGraph avec deux nodes : LLM et exécution des tools
- Le graphe boucle automatiquement jusqu'à la réponse finale

Ce que LangGraph apporte vs exercice 1 :
- La boucle ReAct est gérée automatiquement
- L'état (messages) est géré par LangGraph, pas à la main
- Le graphe est visualisable avec print_ascii()

Pour jouer :
- Affiche le graphe avec agent.get_graph().print_ascii()
- Compare llama3.2:3b et llama3.1:8b sur les mêmes questions
- Ajoute un tool et observe que LangGraph s'adapte sans changer le graphe
- Teste une question très complexe et observe les étapes de raisonnement

Dépendances :
- uv add langgraph langchain-ollama
"""

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
import json

MODEL = ChatOllama(model="llama3.2:3b")


# ── Définition des tools avec @tool ─────────────────────────
@tool
def calculatrice(expression: str) -> str:
    """Effectue des calculs mathématiques. Utilise pour tout calcul numérique."""
    try:
        result = eval(expression)
        return f"Résultat : {result:.2f}"
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


@tool
def conversion_devise(montant: float, de: str, vers: str) -> str:
    """Convertit un montant d'une devise à une autre."""
    taux = {
        ("USD", "EUR"): 0.92,
        ("EUR", "USD"): 1.09,
    }
    key = (de.upper(), vers.upper())
    if key in taux:
        return f"{montant} {de} = {montant * taux[key]:.2f} {vers}"
    return f"Conversion {de} → {vers} non disponible."


# ── Création de l'agent ──────────────────────────────────────
tools = [calculatrice, recherche_etf, conversion_devise]
agent = create_react_agent(MODEL, tools)

# ── Visualiser le graphe ─────────────────────────────────────
print("Structure du graphe :")
agent.get_graph().print_ascii()


# ── Test ─────────────────────────────────────────────────────
def run_agent(question: str):
    print(f"\nQ: {question}\n")
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    print(f"Réponse : {result['messages'][-1].content}")
    print("=" * 60)


questions = [
    "Si j'investis 500€ par mois pendant 20 ans avec 7% de rendement annuel, combien j'aurai ?",
    "Compare le TER du CW8 et du Lyxor MSCI World.",
    "J'ai 10 000 dollars, combien en euros ? Et avec 8% sur 10 ans dans le CW8 ?",
]

for q in questions:
    run_agent(q)
