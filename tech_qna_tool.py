"""
Tool that takes a line of code (or a question about code) as input
and responds with an explanation.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL_GPT = "gpt-4o-mini"
MODEL_LLAMA = "llama3.2"

load_dotenv(override=True)

system_prompt = """
You are a helpful technical tutor who explains how a certain section of code works.
You will explain what the code does and why.
"""

user_prompt_prefix = """
Give a detailed explanation to the following question or code snippet:
"""


def messages_for(question):
    """Create message list for the LLM."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_prefix + question},
    ]


def explain_with_openai(question):
    """Explain the question/code using OpenAI."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=MODEL_GPT,
        messages=messages_for(question),
    )
    return response.choices[0].message.content


def explain_with_ollama(question):
    """Explain the question/code using Ollama (local)."""
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
    response = client.chat.completions.create(
        model=MODEL_LLAMA,
        messages=messages_for(question),
    )
    return response.choices[0].message.content


def main():
    """Main entry point for testing."""
    question = input("Paste your code or question here: ")
    if not question.strip():
        print("No input provided.")
        return

    model_choice = input("Choose model (openai/ollama): ").strip().lower()
    if model_choice == "openai":
        print("\nGetting explanation from OpenAI...\n")
        print(explain_with_openai(question))
    elif model_choice == "ollama":
        print("\nGetting explanation from Ollama...\n")
        print(explain_with_ollama(question))
    else:
        print("Invalid choice. Please enter 'openai' or 'ollama'.")
        return


if __name__ == "__main__":
    main()
