"""
Exercice 3 — RAG avec LangChain (LCEL)

Objectif : refaire le même pipeline avec LangChain et observer ce que
le framework abstrait — et ce qu'il cache.

Ce que fait ce code :
- Charge le document avec un TextLoader LangChain
- Découpe avec RecursiveCharacterTextSplitter
- Génère les embeddings et les stocke dans Chroma via LangChain
- Crée une chain RAG avec LCEL (LangChain Expression Language)
- Pose des questions et récupère les réponses

Ce que LangChain abstrait vs exercices précédents :
- Plus besoin de gérer manuellement les embeddings et le vector store
- La chain LCEL est lisible comme un pipeline — tu vois chaque étape
- Moins de code, mais moins de contrôle — important de savoir ce qui se passe dessous

Pour jouer :
- Change le splitter (CharacterTextSplitter vs RecursiveCharacterTextSplitter)
  et observe la différence sur les chunks générés
- Augmente k dans le retriever et observe l'impact sur la qualité
- Modifie le prompt template et observe comment ça change les réponses
- Remplace StrOutputParser() par un JsonOutputParser pour des réponses structurées
"""

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ── Créer un fichier texte source ────────────────────────────
with open("etf_doc.txt", "w") as f:
    f.write("""
Les ETF (Exchange Traded Funds) sont des fonds d'investissement cotés en bourse.
Ils permettent d'investir dans un panier de valeurs en une seule transaction.
Les ETF répliquent généralement un indice comme le CAC 40 ou le S&P 500.
Leurs frais de gestion sont très faibles, souvent inférieurs à 0.5% par an.
Contrairement aux fonds actifs, les ETF ne cherchent pas à battre le marché.
Un ETF peut être acheté et vendu tout au long de la journée comme une action.
La diversification offerte par les ETF réduit le risque global du portefeuille.
Les ETF distribuants versent des dividendes, les ETF capitalisants les réinvestissent.
Le PEA permet d'investir dans des ETF européens avec une fiscalité avantageuse.
""")

# ── Chargement + Chunking ────────────────────────────────────
loader = TextLoader("etf_doc.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
chunks = splitter.split_documents(documents)
print(f"{len(chunks)} chunks générés\n")

# ── Embeddings + Vector Store ────────────────────────────────
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    chunks, embeddings, persist_directory="./phases/phase3/chroma_langchain"
)

# ── Prompt ───────────────────────────────────────────────────
prompt = ChatPromptTemplate.from_template("""
Voici des extraits de documents :

{context}

En te basant UNIQUEMENT sur ces extraits, réponds à la question suivante :
{question}

Si la réponse n'est pas dans les extraits, dis-le clairement.
""")

# ── Chain LCEL ───────────────────────────────────────────────
llm = Ollama(model="llama3.2:3b")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ── Test ─────────────────────────────────────────────────────
questions = [
    "Quelle est la différence entre un ETF distribuant et capitalisant ?",
    "Comment fonctionne la fiscalité des ETF en France ?",
]

for q in questions:
    print(f"Q: {q}")
    print(f"R: {chain.invoke(q)}\n")
