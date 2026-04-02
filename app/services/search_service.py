import os
from typing import List, Optional
from bs4 import BeautifulSoup
import requests


class SearchService:
    """Service for web search and content fetching using Brave Search API."""

    def __init__(self):
        self.api_key = os.getenv("BRAVE_API_KEY")

        if not self.api_key:
            print("⚠ Warning: BRAVE_API_KEY not set. Search will not work.")

    def search(self, query: str, num_results: int = 10) -> List[dict]:
        """
        Search using Brave Search API and fetch content from top results.

        Returns:
            List of dicts with keys: title, link, content, snippet
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if not self.api_key:
            raise ValueError("Search not configured. Set BRAVE_API_KEY environment variable.")

        # Search using Brave
        search_results = self._brave_search(query, num_results)

        if not search_results:
            raise ValueError("No search results found")

        # Fetch content from top results
        fetched_content = self._fetch_content_from_links(search_results)

        if not fetched_content:
            raise ValueError("Could not fetch content from search results")

        return fetched_content

    def format_results(self, fetched: List[dict]) -> str:
        """Format fetched content for LLM context."""
        formatted = ""
        for i, item in enumerate(fetched, 1):
            formatted += f"\n[Source {i}: {item['title']}]\n"
            formatted += f"URL: {item['link']}\n"
            formatted += f"Content: {item['content']}\n"
            formatted += "-" * 50 + "\n"
        return formatted

    def _brave_search(self, query: str, num_results: int = 10) -> List[dict]:
        """Search using Brave Search API."""
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
            web_data = data.get("web", {})
            results = web_data.get("results", []) if isinstance(web_data, dict) else []

            parsed_results = []
            for item in results:
                try:
                    if not isinstance(item, dict):
                        continue

                    parsed_results.append({
                        "title": item.get("title", ""),
                        "link": item.get("url", ""),
                        "snippet": item.get("description", ""),
                    })
                except Exception as e:
                    print(f"Error parsing result item: {e}")
                    continue

            print(f"✓ Found {len(parsed_results)} results from Brave Search API")
            return parsed_results

        except requests.exceptions.HTTPError as e:
            print(f"✗ Brave API error: {e.response.status_code} - {e.response.text}")
            raise Exception(f"Brave Search API error: {e.response.status_code}")
        except Exception as e:
            print(f"✗ Search error: {e}")
            raise

    def _fetch_content_from_links(self, search_results: List[dict]) -> List[dict]:
        """Fetch and extract text content from URLs."""
        fetched = []

        for result in search_results:
            try:
                content = self._fetch_url_content(result["link"])
                if content:
                    fetched.append({
                        "title": result.get("title"),
                        "link": result.get("link"),
                        "snippet": result.get("snippet"),
                        "content": content[:2000],  # Limit to first 2000 chars
                    })
            except Exception as e:
                print(f"Failed to fetch {result.get('link')}: {e}")
                continue

        return fetched

    def _fetch_url_content(self, url: str) -> Optional[str]:
        """Fetch and extract text content from a URL."""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=5)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            # Remove script and style tags
            for script in soup(["script", "style"]):
                script.decompose()

            # Get text
            text = soup.get_text()
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = " ".join(chunk for chunk in chunks if chunk)

            return text if text else None

        except requests.exceptions.Timeout:
            print(f"⏱ Timeout fetching {url}, skipping...")
            return None
        except requests.exceptions.ConnectionError:
            print(f"🔌 Connection error for {url}, skipping...")
            return None
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                print(f"🚫 Forbidden (403) for {url}, skipping...")
            else:
                print(f"❌ HTTP {e.response.status_code} for {url}, skipping...")
            return None
        except Exception as e:
            print(f"⚠ Error fetching {url}: {str(e)}")
            return None
