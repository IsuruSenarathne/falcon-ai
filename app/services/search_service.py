import os
import time
from typing import List, Optional
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse


class SearchService:
    """Service for web search and content fetching using Brave Search API."""

    # Domains that block automated requests
    BLOCKED_DOMAINS = {
        "shiksha.com",
        "topuniversities.com",
        "mastersportal.com",
        "linkedin.com",
        "facebook.com",
        "twitter.com",
    }

    # Realistic User-Agents
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    ]

    def __init__(self):
        self.api_key = os.getenv("BRAVE_API_KEY")
        self.user_agent_index = 0

        if not self.api_key:
            print("⚠ Warning: BRAVE_API_KEY not set. Search will not work.")

    def search(self, query: str, num_results: int = 10) -> List[dict]:
        """
        Search using Brave Search API and fetch content from top results.

        Returns:
            List of dicts with keys: title, link, content, snippet
        """
        start = time.time()
        print(f"\n{'='*60}")
        print(f"🔍 SearchService.search() starting for: {query[:50]}...")
        print(f"{'='*60}")

        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if not self.api_key:
            raise ValueError("Search not configured. Set BRAVE_API_KEY environment variable.")

        # Search using Brave
        search_start = time.time()
        search_results = self._brave_search(query, num_results)
        search_time = time.time() - search_start
        print(f"⏱️  Brave API call: {search_time:.2f}s")

        if not search_results:
            raise ValueError("No search results found")

        # Fetch content from top results
        fetch_start = time.time()
        fetched_content = self._fetch_content_from_links(search_results)
        fetch_time = time.time() - fetch_start
        print(f"⏱️  Content fetching: {fetch_time:.2f}s")

        if not fetched_content:
            raise ValueError("Could not fetch content from search results")

        total_time = time.time() - start
        print(f"✅ SearchService completed in {total_time:.2f}s")
        print(f"{'='*60}\n")

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
        print(f"  → Calling Brave Search API for: {query[:40]}...")

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

            print(f"    ✓ API returned {len(results)} raw results")

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
                    print(f"    ⚠ Error parsing result item: {e}")
                    continue

            print(f"    ✓ Parsed {len(parsed_results)} results")
            return parsed_results

        except requests.exceptions.HTTPError as e:
            print(f"    ✗ HTTP Error {e.response.status_code}")
            raise Exception(f"Brave Search API error: {e.response.status_code}")
        except Exception as e:
            print(f"    ✗ Search failed: {e}")
            raise

    def _get_headers(self) -> dict:
        """Get realistic headers with rotating User-Agent."""
        user_agent = self.USER_AGENTS[self.user_agent_index % len(self.USER_AGENTS)]
        self.user_agent_index += 1

        return {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "max-age=0",
        }

    def _is_domain_blocked(self, url: str) -> bool:
        """Check if domain is known to block requests."""
        try:
            domain = urlparse(url).netloc.lower()
            # Remove www. prefix
            domain = domain.replace("www.", "")
            return domain in self.BLOCKED_DOMAINS
        except:
            return False

    def _fetch_content_from_links(self, search_results: List[dict]) -> List[dict]:
        """Fetch and extract text content from URLs."""
        print(f"  → Fetching content from {len(search_results)} links...")
        fetched = []

        for i, result in enumerate(search_results, 1):
            url = result.get("link")

            # Skip blocked domains
            if self._is_domain_blocked(url):
                print(f"    [{i}/{len(search_results)}] Skipping blocked domain: {url[:50]}...")
                continue

            try:
                print(f"    [{i}/{len(search_results)}] Fetching: {url[:50]}...")
                content = self._fetch_url_content(url)
                if content:
                    fetched.append({
                        "title": result.get("title"),
                        "link": result.get("link"),
                        "snippet": result.get("snippet"),
                        "content": content
                    })
                    print(f"          ✓ Success ({len(content)} chars)")
                else:
                    print(f"          ⚠ No content extracted")
            except Exception as e:
                print(f"          ✗ Failed: {str(e)[:50]}")
                continue

        print(f"    ✓ Successfully fetched {len(fetched)}/{len(search_results)} links")
        return fetched

    def _fetch_url_content(self, url: str, retry_count: int = 0, max_retries: int = 2) -> Optional[str]:
        """Fetch and extract text content from a URL with retry logic."""
        try:
            headers = self._get_headers()
            response = requests.get(url, headers=headers, timeout=5)
            response.raise_for_status()

            # Use lxml parser for speed
            try:
                soup = BeautifulSoup(response.content, "lxml")
            except:
                soup = BeautifulSoup(response.content, "html.parser")

            # Remove non-content elements
            for element in soup(["script", "style", "nav", "header", "footer", "noscript", "meta", "link"]):
                element.decompose()

            # Also remove common ad/tracking elements
            for element in soup.find_all(class_=lambda x: x and any(ad_word in x.lower() for ad_word in ["ad", "advertisement", "sponsored", "sidebar"])):
                element.decompose()

            # Get text from main content areas if available
            text = ""
            main_content = soup.find(["main", "article", "div"], class_=lambda x: x and "content" in x.lower())

            if main_content:
                text = main_content.get_text(separator=" ", strip=True)
            else:
                # Fallback to body text
                body = soup.find("body")
                if body:
                    text = body.get_text(separator=" ", strip=True)
                else:
                    text = soup.get_text(separator=" ", strip=True)

            # Clean up whitespace
            lines = (line.strip() for line in text.split("\n") if line.strip())
            text = " ".join(lines)

            # Remove multiple spaces
            text = " ".join(text.split())

            return text if text else None

        except requests.exceptions.Timeout:
            if retry_count < max_retries:
                wait_time = 2 ** retry_count  # Exponential backoff: 1s, 2s, 4s
                print(f"⏱ Timeout, retrying in {wait_time}s ({retry_count + 1}/{max_retries})...")
                time.sleep(wait_time)
                return self._fetch_url_content(url, retry_count + 1, max_retries)
            print(f"⏱ Timeout fetching {url} after {max_retries} retries, skipping...")
            return None

        except requests.exceptions.ConnectionError:
            if retry_count < max_retries:
                wait_time = 2 ** retry_count
                print(f"🔌 Connection error, retrying in {wait_time}s ({retry_count + 1}/{max_retries})...")
                time.sleep(wait_time)
                return self._fetch_url_content(url, retry_count + 1, max_retries)
            print(f"🔌 Connection error for {url} after {max_retries} retries, skipping...")
            return None

        except requests.exceptions.HTTPError as e:
            # 403 is permanent, don't retry
            if e.response.status_code == 403:
                print(f"🚫 Forbidden (403), skipping...")
                return None
            # 429 is rate limit, retry
            elif e.response.status_code == 429:
                if retry_count < max_retries:
                    wait_time = 3 * (2 ** retry_count)  # Longer wait for rate limits
                    print(f"⚠ Rate limited, retrying in {wait_time}s ({retry_count + 1}/{max_retries})...")
                    time.sleep(wait_time)
                    return self._fetch_url_content(url, retry_count + 1, max_retries)
                print(f"⚠ Rate limited after {max_retries} retries, skipping...")
                return None
            # Other HTTP errors
            else:
                print(f"❌ HTTP {e.response.status_code}, skipping...")
                return None

        except Exception as e:
            print(f"⚠ Error fetching {url}: {str(e)[:50]}")
            return None
