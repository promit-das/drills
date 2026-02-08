"""
Ingest the knowledge base into a local Chroma vector store using Hugging Face embeddings.
"""


#imports
import os
import shutil
from pathlib import Path
import tiktoken
import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.manifold import TSNE
import plotly.graph_objects as go


# models
MODEL = "gpt-4.1-nano"
BASE_DIR = Path(__file__).resolve().parent
KB_DIR = BASE_DIR / "knowledge-base"
DB_DIR = BASE_DIR / "vector_db"
load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")


# Load in everything in the knowledgebase using LangChain's loaders
if not KB_DIR.exists():
    raise FileNotFoundError(f"Knowledge base directory not found: {KB_DIR}")

folders = [p for p in KB_DIR.iterdir() if p.is_dir()]

documents = []
for folder in folders:
    doc_type = os.path.basename(folder)
    loader = DirectoryLoader(
        str(folder),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8', 'autodetect_encoding': True},
    )
    folder_docs = loader.load()
    for doc in folder_docs:
        doc.metadata["doc_type"] = doc_type
        documents.append(doc)


# Divide into chunks using the RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)


# Pick an embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

if DB_DIR.exists():
    try:
        Chroma(persist_directory=str(DB_DIR), embedding_function=embeddings).delete_collection()
    except Exception as exc:
        print(f"Warning: failed to delete existing Chroma collection: {exc}")
        shutil.rmtree(DB_DIR)

if not documents:
    raise ValueError(f"No documents found under {KB_DIR}. Check your knowledge base paths.")

vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=str(DB_DIR))
