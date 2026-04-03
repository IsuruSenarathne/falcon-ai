"""Simple web search service using Brave Search API."""
import os
from typing import List, Optional
from bs4 import BeautifulSoup
import requests


class SearchService:
    """Web search and content fetching service."""

    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    ]

    def __init__(self):
        self.api_key = os.getenv("BRAVE_API_KEY")
        self.user_agent_index = 0

    def search(self, query: str, num_results: int = 5) -> List[dict]:
        """
        Search using Brave Search API and fetch content from top results.

        Args:
            query: Search query string
            num_results: Number of results to fetch (default 5)

        Returns:
            List of dicts with keys: title, link, content
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if not self.api_key:
            raise ValueError("BRAVE_API_KEY not set")

        # Search using Brave API
        search_results = self._brave_search(query, num_results)

        if not search_results:
            return []

        # Fetch content from top results
        fetched = []
        for result in search_results:
            content = self._fetch_url_content(result.get("link"))
            if content:
                fetched.append({
                    "title": result.get("title"),
                    "link": result.get("link"),
                    "content": content
                })

        return fetched

    def _brave_search(self, query: str, num_results: int = 5) -> List[dict]:
        """Call Brave Search API."""
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key,
        }
        params = {
            "q": query,
            "count": num_results,
        }

        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            results = data.get("web", {}).get("results", [])

            parsed_results = []
            for item in results:
                if isinstance(item, dict):
                    parsed_results.append({
                        "title": item.get("title", ""),
                        "link": item.get("url", ""),
                    })

            return parsed_results

        except Exception as e:
            raise Exception(f"Brave Search API error: {e}")

    def _fetch_url_content(self, url: str) -> Optional[str]:
        """Fetch and extract text content from a URL."""
        try:
            headers = self._get_headers()
            response = requests.get(url, headers=headers, timeout=5)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            # Remove script and style elements
            for element in soup(["script", "style"]):
                element.decompose()

            # Get text
            text = soup.get_text(separator=" ", strip=True)
            text = " ".join(text.split())

            return text if text else None

        except Exception:
            return None

    def _get_headers(self) -> dict:
        """Get headers with rotating User-Agent."""
        user_agent = self.USER_AGENTS[self.user_agent_index % len(self.USER_AGENTS)]
        self.user_agent_index += 1

        return {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
