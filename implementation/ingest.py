"""
Ingest the knowledge base into a Chroma vector store for later retrieval.
"""

from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent
DB_NAME = BASE_DIR / "vector_db_implementation"
KNOWLEDGE_BASE = BASE_DIR / "knowledge-base"

load_dotenv(override=True)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


def fetch_documents():
    base = KNOWLEDGE_BASE
    documents = []
    for folder in base.iterdir():
        if not folder.is_dir():
            continue
        loader = DirectoryLoader(
            str(folder),
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        for doc in loader.load():
            doc.metadata["doc_type"] = folder.name
            documents.append(doc)
    return documents


def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    return text_splitter.split_documents(documents)


def create_embeddings(chunks):
    if DB_NAME.exists():
        Chroma(persist_directory=str(DB_NAME), embedding_function=embeddings).delete_collection()

    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=str(DB_NAME)
    )

    collection = vectorstore._collection
    count = collection.count()

    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)
    print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")
    return vectorstore


if __name__ == "__main__":
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_embeddings(chunks)
    print("Ingestion complete")
