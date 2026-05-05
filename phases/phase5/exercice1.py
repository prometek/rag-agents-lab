"""
Exercice 1 — Tool use natif Ollama

Objectif : comprendre comment un LLM local décide quand appeler un tool
et comment gérer la boucle de conversation sans API externe.

Ce que fait ce code :
- Définit 3 tools : calculatrice, recherche ETF, conversion devise
- Lance une boucle agent : le modèle décide quoi appeler, on exécute, on renvoie le résultat
- Continue jusqu'à ce que le modèle ait une réponse finale

Ce que tu observes :
- Le modèle choisit lui-même quel tool appeler selon la question
- Il peut enchaîner plusieurs tools dans la même conversation
- Le docstring de chaque tool est ce que le modèle lit pour décider

Pour jouer :
- Ajoute un nouveau tool (ex: calcul de performance sur X années)
- Pose des questions qui nécessitent plusieurs tools enchaînés
- Modifie la description d'un tool et observe si le modèle l'utilise différemment
- Pose une question hors scope et vois comment le modèle réagit sans tool adapté
- Compare llama3.2:3b et llama3.1:8b sur les mêmes questions
"""

import ollama
import json

MODEL = "llama3.2:3b"

# ── Définition des tools ─────────────────────────────────────
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculatrice",
            "description": "Effectue des calculs mathématiques simples. Utilise cet outil pour tout calcul numérique.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "L'expression mathématique à calculer, ex: '1000 * 1.08 ** 10'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recherche_etf",
            "description": "Recherche les informations d'un ETF par son nom : TER, encours, indice répliqué, éligibilité PEA.",
            "parameters": {
                "type": "object",
                "properties": {
                    "nom": {
                        "type": "string",
                        "description": "Le nom ou ticker de l'ETF, ex: 'Amundi MSCI World' ou 'CW8'",
                    }
                },
                "required": ["nom"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "conversion_devise",
            "description": "Convertit un montant d'une devise à une autre.",
            "parameters": {
                "type": "object",
                "properties": {
                    "montant": {
                        "type": "number",
                        "description": "Le montant à convertir",
                    },
                    "de": {"type": "string", "description": "Devise source, ex: 'USD'"},
                    "vers": {
                        "type": "string",
                        "description": "Devise cible, ex: 'EUR'",
                    },
                },
                "required": ["montant", "de", "vers"],
            },
        },
    },
]


# ── Implémentation des tools ─────────────────────────────────
def calculatrice(expression: str) -> str:
    try:
        result = eval(expression)
        return f"Résultat : {result:.2f}"
    except Exception as e:
        return f"Erreur de calcul : {e}"


def recherche_etf(nom: str) -> str:
    etfs = {
        "amundi msci world": {
            "nom": "Amundi MSCI World",
            "ticker": "CW8",
            "ter": "0.38%",
            "pea": "Oui",
            "type": "Capitalisant",
        },
        "cw8": {
            "nom": "Amundi MSCI World",
            "ticker": "CW8",
            "ter": "0.38%",
            "pea": "Oui",
            "type": "Capitalisant",
        },
        "lyxor msci world": {
            "nom": "Lyxor MSCI World",
            "ticker": "EWLD",
            "ter": "0.12%",
            "pea": "Non",
            "type": "Distribuant",
        },
    }
    key = nom.lower()
    if key in etfs:
        return json.dumps(etfs[key], ensure_ascii=False)
    return f"ETF '{nom}' non trouvé."


def conversion_devise(montant: float, de: str, vers: str) -> str:
    taux = {
        ("USD", "EUR"): 0.92,
        ("EUR", "USD"): 1.09,
        ("GBP", "EUR"): 1.17,
        ("EUR", "GBP"): 0.85,
    }
    key = (de.upper(), vers.upper())
    if key in taux:
        return f"{montant} {de} = {montant * taux[key]:.2f} {vers}"
    return f"Conversion {de} → {vers} non disponible."


def execute_tool(name: str, inputs: dict) -> str:
    if name == "calculatrice":
        return calculatrice(inputs.get("expression", ""))
    elif name == "recherche_etf":
        return recherche_etf(inputs.get("nom", ""))
    elif name == "conversion_devise":
        return conversion_devise(
            montant=float(inputs.get("montant", 0)),
            de=inputs.get("de", ""),
            vers=inputs.get("vers", ""),
        )
    return f"Tool '{name}' inconnu."


# ── Boucle agent ─────────────────────────────────────────────
def run_agent(question: str):
    print(f"\nQ: {question}\n")
    messages = [{"role": "user", "content": question}]

    while True:
        response = ollama.chat(model=MODEL, messages=messages, tools=tools)

        # Pas de tool appelé → réponse finale
        if not response.message.tool_calls:
            print(f"Réponse : {response.message.content}")
            break

        # Ajouter la réponse du modèle à l'historique
        messages.append(response.message)

        # Exécuter tous les tools demandés
        for tool_call in response.message.tool_calls:
            name = tool_call.function.name
            inputs = tool_call.function.arguments
            print(f"→ Tool appelé : {name}({inputs})")
            result = execute_tool(name, inputs)
            print(f"→ Résultat    : {result}\n")

            messages.append({"role": "tool", "content": result})


# ── Test ─────────────────────────────────────────────────────
questions = [
    "Si j'investis 500€ par mois pendant 20 ans avec un rendement annuel de 7%, combien j'aurai ?",
    "Compare le TER du CW8 et du Lyxor MSCI World. Lequel est le moins cher ?",
    "J'ai 10 000 dollars, combien ça fait en euros ? Et si je les investis dans le CW8 avec 8% sur 10 ans ?",
]

for q in questions:
    run_agent(q)
    print("=" * 60)
