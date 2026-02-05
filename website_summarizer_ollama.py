"""
Website summarizer using Ollama instead of OpenAI.
"""

from openai import OpenAI
from scraper import fetch_website_contents

OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL = "llama3.2"

system_prompt = """
You are a snarky assistant that analyzes the contents of a website.
Provide a short, snarky, humorous summary, ignoring text that might be navigation related.
"""

user_prompt_prefix = """
Here are the contents of a website.
Provide a short summary of this website.
If it includes news or announcements, then summarize these too.
"""


def messages_for(website):
    """Create message list for the LLM."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_prefix + website},
    ]


def summarize(url):
    """Fetch and summarize a website using Ollama."""
    if not url.strip():
        raise ValueError("URL is required.")

    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
    website = fetch_website_contents(url)
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages_for(website),
    )
    return response.choices[0].message.content


def main():
    url = input("Enter a URL to summarize: ").strip()
    if not url:
        print("No URL provided.")
        return

    print("\nFetching and summarizing...\n")
    try:
        summary = summarize(url)
    except Exception as exc:
        print(f"Error: {exc}")
        return
    print(summary)


if __name__ == "__main__":
    main()
