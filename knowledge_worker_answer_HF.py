"""
Serve a Gradio chat UI that answers questions using a Chroma-backed retriever and an LLM.
"""


import os
from pathlib import Path

# imports
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
import gradio as gr


# constants
MODEL = "gpt-4.1-nano"
BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "vector_db"
load_dotenv(override=True)

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not set. Set it before running this script.")


# connect to Chroma
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
if not DB_DIR.exists():
    raise FileNotFoundError(
        f"Vector store not found at {DB_DIR}. Run knowledge_worker_ingest_HF.py first."
    )
vectorstore = Chroma(persist_directory=str(DB_DIR), embedding_function=embeddings)


# setup LangChain objects: retreiver & llm
retriever = vectorstore.as_retriever()
llm = ChatOpenAI(temperature=0, model_name=MODEL)
"""
Temperature controls which tokens get selected during inference
- temperature=0 always selects the token with highest probability
- temperature=1 means that a token with 10% probability should be picked 10% of the time
"""


# system prompt
SYSTEM_PROMPT_TEMPLATE = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""


# RAG
def answer_question(question: str, history):
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=question)])
    return response.content


# Gradio chat interface
gr.ChatInterface(answer_question).launch(inbrowser=True)
