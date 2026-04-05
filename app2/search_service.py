"""Simple web search service using Brave Search API."""
import os
import logging
from typing import List, Optional
from bs4 import BeautifulSoup
import requests

logger = logging.getLogger(__name__)


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
        logger.info(f"  → Calling Brave Search API...")
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
            logger.debug(f"  → Making request to Brave API...")
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            results = data.get("web", {}).get("results", [])
            logger.info(f"    ✓ Got {len(results)} raw results from API")

            parsed_results = []
            for item in results:
                if isinstance(item, dict):
                    parsed_results.append({
                        "title": item.get("title", ""),
                        "link": item.get("url", ""),
                    })

            logger.info(f"    ✓ Parsed {len(parsed_results)} results")
            return parsed_results

        except Exception as e:
            logger.error(f"    ✗ Brave Search API error: {e}")
            raise Exception(f"Brave Search API error: {e}")

    def _fetch_url_content(self, url: str) -> Optional[str]:
        """Fetch and extract text content from a URL."""
        try:
            logger.debug(f"    Fetching content from: {url[:60]}...")
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

            if text:
                logger.debug(f"      ✓ Extracted {len(text)} characters")
                return text
            else:
                logger.debug(f"      ⚠ No content found")
                return None

        except Exception as e:
            logger.debug(f"      ✗ Failed to fetch: {str(e)[:50]}")
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
