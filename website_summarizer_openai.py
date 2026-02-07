"""
Website summarizer using OpenAI and the local scraper helpers.
"""

import os
from dotenv import load_dotenv
from scraper import fetch_website_contents
from openai import OpenAI

MODEL = "gpt-4o-mini"

load_dotenv(override=True)

# system prompt
system_prompt = """
You are a helpful assistant that analyzes the contents of a website.
Provide a short, to-the-point, humorous summary of the contents.
Ignore text that might be navigation related.
Do not wrap the markdown in a code block. Respond just with the markdown.
"""

# user prompt prefix
user_prompt_prefix = """
Here are the contents of a website.
Provide a short summary of this website.
Summarize any news or announcements in this website.
"""

# messages for the LLM
def messages_for(website):
    """Create message list for the LLM."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_prefix + website},
    ]

# summarize the website
def summarize(url):
    """Fetch and summarize a website using OpenAI."""
    if not url.strip():
        raise ValueError("URL is required.")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)
    website = fetch_website_contents(url)
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages_for(website),
    )
    return response.choices[0].message.content

# main entry point for testing
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
