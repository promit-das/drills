# Proof of Concept

## Contents

- `scraper.py`: Fetches a web page with `requests` and extracts readable text/links using BeautifulSoup, with basic cleanup and truncation.
- `tech_qna_tool.py`: CLI tool that sends a code snippet or question to a selected LLM (OpenAI or local Ollama) and prints the explanation.
- `website_summarizer_openai.py`: CLI website summarizer that uses OpenAI to produce a short, humorous summary of page content.
- `website_summarizer_ollama.py`: CLI website summarizer that uses local Ollama to produce a short, snarky summary of page content.

## Quick Setup

1. Install `uv` (see [uv docs](https://docs.astral.sh/uv/)) and confirm: `uv --version`.
2. Install deps: `uv sync`.
3. Add secrets: create `.env` in repo root with:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
   Keep `.env` private and uncommitted.
   
