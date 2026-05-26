import requests
from crewai.tools import tool
from ddgs import DDGS


@tool("DuckDuckGo Search")
def duckduckgo_search(query: str) -> str:
    """Search the web using DuckDuckGo and return the top results.
    Use this to find official websites for tourist attractions.
    Input should be a search query string, e.g. 'Eiffel Tower official website'.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return "No results found."
        return "\n\n".join(
            f"Title: {r.get('title', '')}\nURL: {r.get('href', '')}\nSnippet: {r.get('body', '')}"
            for r in results
        )
    except Exception as e:
        return f"Search failed: {e}"


@tool("URL Validator")
def validate_url(url: str) -> str:
    """Check whether a URL is reachable and returns a successful HTTP response.
    Returns 'valid' if the site responds with HTTP < 400, otherwise 'invalid: <reason>'.
    Use this to confirm an attraction's website actually exists before including it.
    """
    try:
        resp = requests.get(
            url,
            timeout=8,
            allow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; TripAdvisorBot/1.0)"},
        )
        if resp.status_code < 400:
            return "valid"
        return f"invalid: HTTP {resp.status_code}"
    except requests.exceptions.Timeout:
        return "invalid: request timed out"
    except requests.exceptions.ConnectionError:
        return "invalid: could not connect"
    except Exception as e:
        return f"invalid: {e}"
