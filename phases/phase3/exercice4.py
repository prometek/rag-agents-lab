"""
Exercice 4 — Chat avec un PDF

Objectif : premier projet concret — charger un vrai PDF, l'indexer,
et pouvoir poser des questions dessus en conversation.

Ce que fait ce code :
- Charge un PDF avec PyPDFLoader
- Découpe, indexe dans ChromaDB via LangChain
- Lance une boucle de conversation interactive
- Maintient l'historique de la conversation pour les questions de suivi

Pour jouer :
- Remplace le PDF par n'importe quel document (rapport, contrat, doc technique)
- Pose des questions de suivi qui font référence aux réponses précédentes
- Teste avec un PDF long (50+ pages) et observe les performances
- Change le modèle Ollama pour comparer la qualité des réponses
- Modifie le prompt pour forcer une réponse en bullet points

Prérequis :
- Avoir un fichier PDF dans le même dossier, nommé "document.pdf"
- uv add pypdf
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# ── Chargement PDF ───────────────────────────────────────────
loader = PyPDFLoader("phases/phase3/document.pdf")
pages = loader.load()
print(f"{len(pages)} pages chargées\n")

# ── Chunking ─────────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(pages)
print(f"{len(chunks)} chunks générés\n")

# ── Embeddings + Vector Store ────────────────────────────────
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    chunks, embeddings, persist_directory="./phases/phase3/chroma_pdf"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ── Prompt avec historique ───────────────────────────────────
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Tu es un assistant qui répond aux questions sur un document.
Utilise UNIQUEMENT les extraits suivants pour répondre :

{context}

Si la réponse n'est pas dans les extraits, dis-le clairement.""",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

# ── Chain LCEL ───────────────────────────────────────────────
llm = Ollama(model="llama3.2:3b")

chain = (
    {
        "context": lambda x: retriever.invoke(x["question"]),
        "question": lambda x: x["question"],
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    | llm
    | StrOutputParser()
)

# ── Boucle de conversation ───────────────────────────────────
print("Chat avec ton PDF — tape 'quit' pour quitter\n")

chat_history = []

while True:
    question = input("Toi : ")
    if question.lower() == "quit":
        break

    response = chain.invoke({"question": question, "chat_history": chat_history})

    print(f"Agent : {response}\n")

    # Mise à jour de l'historique
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=response))
