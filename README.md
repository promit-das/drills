# Proof of Concept

## Contents

- `brochure_generator.py`: Scrapes a company site, selects relevant links with OpenAI, and streams a short brochure to stdout.
- `code_porter.py`: Gradio UI to paste Python code and port it to high-performance C++ or Rust via LLM backends.
- `app_implementation.py`: Gradio UI for the Insurellm assistant wired to the implementation RAG pipeline.
- `knowledge_worker_ingest_HF.py`: Ingests the knowledge base into a Chroma vector store using Hugging Face embeddings.
- `knowledge_worker_answer_HF.py`: Gradio chat UI that answers questions using a Chroma retriever and LLM.
- `mom_generator_CLI.py`: CLI tool to transcribe audio (OpenAI or HF) and generate minutes of meeting via OpenAI or Ollama.
- `mom_generator_UI.py`: Gradio UI for transcription and MoM generation, reusing the same providers as the CLI.
- `multi_modal.py`: Gradio demo for a travel assistant with tool calls, TTS, and image generation.
- `scraper.py`: Lightweight helpers to fetch page text and links with requests + BeautifulSoup.
- `styles.py`: CSS theme helpers for Gradio apps.
- `system_info.py`: Utility functions to collect OS, CPU, and toolchain details for LLM guidance.
- `tech_qna_tool_CLI.py`: CLI that explains code or questions via OpenAI or local Ollama.
- `tech_qna_tool_UI.py`: Gradio UI that streams code explanations via OpenAI or Ollama.
- `website_summarizer_openai.py`: CLI website summarizer using OpenAI and the local scraper helpers.
- `website_summarizer_ollama.py`: CLI website summarizer using Ollama and the local scraper helpers.
- `implementation/answer.py`: RAG helper functions to retrieve context and generate answers.
- `implementation/ingest.py`: Ingests the knowledge base into a Chroma vector store for the implementation pipeline.

## Quick Setup

1. Install `uv` (see [uv docs](https://docs.astral.sh/uv/)) and confirm: `uv --version`.
2. Install deps: `uv sync`.
3. Add secrets: create `.env` in repo root with:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
   Keep `.env` private and uncommitted.
   
