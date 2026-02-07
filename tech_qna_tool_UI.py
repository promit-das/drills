"""
Gradio UI that takes a line of code (or a prompt about code) as input
and streams an explanation using OpenAI or Ollama.
"""

import os

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL_LLAMA = "llama3.2"
MODEL_OPENAI = "gpt-4o-mini"

SYSTEM_PROMPT = (
    "You are a helpful technical tutor who explains how a certain section of "
    "code works. You will explain what the code does and why."
)
USER_PROMPT_PREFIX = "Give a detailed explanation to the following prompt or code snippet:\n"

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
ollama_client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")


def messages_for(prompt: str) -> list[dict[str, str]]:
    """Create message list for the LLM."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_PREFIX + prompt},
    ]


def stream_with_openai(prompt: str):
    """Stream explanation using OpenAI."""
    if not openai_client:
        raise ValueError("OPENAI_API_KEY is not set. Add it to .env to use OpenAI.")
    response = openai_client.chat.completions.create(
        model=MODEL_OPENAI,
        messages=messages_for(prompt),
        stream=True,
    )
    for chunk in response:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            yield delta


def stream_with_ollama(prompt: str):
    """Stream explanation using Ollama (local)."""
    response = ollama_client.chat.completions.create(
        model=MODEL_LLAMA,
        messages=messages_for(prompt),
        stream=True,
    )
    for chunk in response:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            yield delta


def explain_prompt(prompt: str, model: str):
    if not prompt or not prompt.strip():
        yield "Please paste a prompt or code snippet."
        return
    if model == "OpenAI":
        stream = stream_with_openai(prompt)
    elif model == "Ollama":
        stream = stream_with_ollama(prompt)
    else:
        raise ValueError(f"Unknown model: {model}")

    buffer = ""
    for delta in stream:
        buffer += delta
        yield buffer


css = """
#explain-btn {
  transition: background-color 0.15s ease, border-color 0.15s ease, color 0.15s ease;
}
#explain-btn:active,
#explain-btn[disabled] {
  background-color: #0b5cff;
  border-color: #0b5cff;
  color: #ffffff;
}
"""

with gr.Blocks(css=css) as view:
    gr.Markdown("# Code Explanation Tool")
    message_input = gr.Textbox(
        label="Paste your code or prompt here",
        info="Enter a message for the LLM",
        lines=7,
    )
    model_selector = gr.Radio(
        ["OpenAI", "Ollama"],
        label="Model",
        value="OpenAI",
    )
    run_button = gr.Button("Explain", elem_id="explain-btn")
    message_output = gr.Markdown(label="Response")

    run_button.click(
        fn=explain_prompt,
        inputs=[message_input, model_selector],
        outputs=message_output,
    )

view.launch(inbrowser=True)
