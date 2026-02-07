"""
Generate a short company brochure from a website.

Scrapes the landing page plus relevant pages, uses OpenAI to choose links,
then streams a brochure to stdout.
"""

# imports
import os
import json
from urllib.parse import urljoin, urlparse
from dotenv import load_dotenv
from scraper import fetch_website_links, fetch_website_contents
from openai import OpenAI

# Initialize and constants
load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')

if api_key and api_key.startswith('sk-proj-') and len(api_key)>10:
    print("API key looks good so far")
else:
    print("There might be a problem with your API key? Please visit the troubleshooting notebook!")
    
MODEL = "gpt-5-nano"
openai = OpenAI()

link_system_prompt = """
You are provided with a list of links found on a webpage.
You are able to decide which of the links would be most relevant to include in a brochure about the company,
such as links to an About page, or a Company page, or Careers/Jobs pages.
You must return absolute https URLs. If a link is relative (e.g., "/about"), resolve it against the base URL.
You should respond in JSON as in this example:

{
    "links": [
        {"type": "about page", "url": "https://full.url/goes/here/about"},
        {"type": "careers page", "url": "https://another.full.url/careers"}
    ]
}
"""

def get_links_user_prompt(url):
    user_prompt = f"""
Here is the list of links on the website {url} -
Please decide which of these are relevant web links for a brochure about the company, 
respond with the full https URL in JSON format.
If a link is relative, resolve it against {url}.
Do not include Terms of Service, Privacy, email links.

Links (some might be relative links):

"""
    links = fetch_website_links(url)
    user_prompt += "\n".join(links)
    return user_prompt


def _normalize_url(base_url, candidate_url):
    if not candidate_url:
        return None
    candidate_url = candidate_url.strip()
    if candidate_url.startswith(("mailto:", "tel:", "javascript:", "#")):
        return None
    # Handle scheme-relative URLs like //example.com/about
    if candidate_url.startswith("//"):
        candidate_url = "https:" + candidate_url
    full_url = urljoin(base_url, candidate_url)
    parsed = urlparse(full_url)
    if not parsed.scheme:
        full_url = "https://" + full_url.lstrip("/")
        parsed = urlparse(full_url)
    if parsed.scheme not in ("http", "https"):
        return None
    return full_url


def select_relevant_links(url):
    print(f"Selecting relevant links for {url} by calling {MODEL}")
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": link_system_prompt},
            {"role": "user", "content": get_links_user_prompt(url)}
        ],
        response_format={"type": "json_object"}
    )
    result = response.choices[0].message.content
    try:
        parsed = json.loads(result)
    except json.JSONDecodeError:
        print("Model returned invalid JSON; falling back to empty link set.")
        return {"links": []}
    raw_links = parsed.get("links", [])
    normalized = []
    seen = set()
    for link in raw_links:
        link_type = (link or {}).get("type", "").strip() or "relevant page"
        link_url = (link or {}).get("url", "").strip()
        full_url = _normalize_url(url, link_url)
        if full_url and full_url not in seen:
            seen.add(full_url)
            normalized.append({"type": link_type, "url": full_url})
    print(f"Found {len(normalized)} relevant links")
    return {"links": normalized}

def fetch_page_and_all_relevant_links(url):
    contents = fetch_website_contents(url)
    relevant_links = select_relevant_links(url)
    result = f"## Landing Page:\n\n{contents}\n## Relevant Links:\n"
    for link in relevant_links['links']:
        result += f"\n\n### Link: {link['type']}\n"
        result += fetch_website_contents(link["url"])
    return result

brochure_system_prompt = """
You are an assistant that analyzes the contents of several relevant pages from a company website
and creates a short brochure about the company for prospective customers, investors and recruits.
Respond in markdown without code blocks.
Include details of company culture, customers and careers/jobs if you have the information.
"""

# Or uncomment the lines below for a more humorous brochure - this demonstrates how easy it is to incorporate 'tone':

# brochure_system_prompt = """
# You are an assistant that analyzes the contents of several relevant pages from a company website
# and creates a short, humorous, entertaining, witty brochure about the company for prospective customers, investors and recruits.
# Respond in markdown without code blocks.
# Include details of company culture, customers and careers/jobs if you have the information.
# """

def get_brochure_user_prompt(company_name, url):
    user_prompt = f"""
You are looking at a company called: {company_name}
Here are the contents of its landing page and other relevant pages;
use this information to build a short brochure of the company in markdown without code blocks.\n\n
"""
    user_prompt += fetch_page_and_all_relevant_links(url)
    user_prompt = user_prompt[:5_000] # Truncate if more than 5,000 characters
    return user_prompt


def stream_brochure(company_name, url):
    stream = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": brochure_system_prompt},
            {"role": "user", "content": get_brochure_user_prompt(company_name, url)}
          ],
        stream=True
    )    
    for chunk in stream:
        print(chunk.choices[0].delta.content or "", end="", flush=True)

def main():
    """Main entry point for testing."""
    url = input("input your URL here: ").strip()
    if not url:
        print("No input provided.")
        return
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    company_name = input("Company name (optional, press enter to infer): ").strip()
    if not company_name:
        parsed = urlparse(url)
        company_name = parsed.hostname.replace("www.", "").split(".")[0].title() if parsed.hostname else "Company"
    stream_brochure(company_name, url)
    print()

if __name__ == "__main__":
    main()
