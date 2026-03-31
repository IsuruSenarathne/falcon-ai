import os
import time
from typing import List, Optional
from bs4 import BeautifulSoup
import requests

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.dto.conversation_dto import SearchRequest, SearchResponse
from app.models.conversation import MessageStatus
from app.services.conversation_service import ConversationService


class SearchService:
    """Service for web search + RAG-based answering using Brave Search API."""

    def __init__(self):
        self.api_key = os.getenv("BRAVE_API_KEY")

        if not self.api_key:
            print("⚠ Warning: BRAVE_API_KEY not set. Search will not work.")

        # LLM for synthesizing answers from fetched content
        template = """Based on the following web search results, answer this question:

Question: {question}

Search Results:
{content}

Provide a comprehensive answer using ONLY information from the search results above.
Format your answer in HTML.

Respond in two sections:
ANSWER:
<p>Your answer here...</p>

REASONING:
<p>Your reasoning here...</p>
"""

        self.answer_chain = (
            ChatPromptTemplate.from_template(template)
            | ChatOllama(model="llama3.2:1b")
            | StrOutputParser()
        )

    def search(self, req: SearchRequest) -> SearchResponse:
        """Execute web search and generate answer from results."""
        if not req.question or not req.question.strip():
            raise ValueError("Question cannot be empty")

        if not self.api_key:
            raise ValueError("Search not configured. Set BRAVE_API_KEY environment variable.")

        start = time.time()
        try:
            # Search using Brave
            search_results = self._brave_search(req.question)
            if not search_results:
                raise ValueError("No search results found")

            # Fetch content from top 5 links
            fetched_content = self._fetch_content_from_links(search_results[:5])
            if not fetched_content:
                raise ValueError("Could not fetch content from search results")

            # Use LLM to synthesize answer from fetched content
            combined_content = self._format_search_results(fetched_content)
            raw_response = self.answer_chain.invoke({
                "question": req.question,
                "content": combined_content
            })

            response_time = time.time() - start

            # Parse response into answer and reasoning
            from app.services.rag_service import RAGService
            rag_service = RAGService.__new__(RAGService)  # Create dummy instance for parsing
            answer, reasoning = rag_service._parse_response(raw_response)
            formatted_answer = rag_service._format_with_reasoning(answer, reasoning)

            # Save to database
            conversation_id, bot_msg = ConversationService.save_exchange(
                question=req.question,
                answer=formatted_answer,
                status=MessageStatus.SUCCESS,
                user_id=req.user_id,
                session_id=req.session_id,
                response_time=response_time,
            )

            return SearchResponse(
                conversation_id=conversation_id,
                question=req.question,
                answer=formatted_answer,
                status="success",
                response_time=response_time,
                created_at=bot_msg.created_at.isoformat(),
                sources=[r.get("link") for r in fetched_content],
            )

        except ValueError:
            raise
        except Exception as e:
            response_time = time.time() - start

            conversation_id, _ = ConversationService.save_exchange(
                question=req.question,
                answer=None,
                status=MessageStatus.ERROR,
                error=str(e),
                user_id=req.user_id,
                session_id=req.session_id,
                response_time=response_time,
            )

            return SearchResponse(
                conversation_id=conversation_id,
                question=req.question,
                answer=None,
                status="error",
                response_time=response_time,
                created_at="",
                error=str(e),
                sources=[],
            )

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
            print(f"Brave API response keys: {data.keys()}")

            # Brave returns results in 'web' -> 'results'
            web_data = data.get("web", {})
            results = web_data.get("results", []) if isinstance(web_data, dict) else []
            print(f"Raw results type: {type(results)}, count: {len(results)}")

            # Parse results
            parsed_results = []
            for item in results:
                try:
                    if not isinstance(item, dict):
                        print(f"Warning: item is {type(item)}, skipping: {item}")
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
            response = requests.get(url, headers=headers, timeout=10)
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

        except Exception as e:
            raise Exception(f"Failed to fetch {url}: {str(e)}")

    def _format_search_results(self, fetched: List[dict]) -> str:
        """Format fetched content for LLM context."""
        formatted = ""
        for i, item in enumerate(fetched, 1):
            formatted += f"\n[Source {i}: {item['title']}]\n"
            formatted += f"URL: {item['link']}\n"
            formatted += f"Content: {item['content']}\n"
            formatted += "-" * 50 + "\n"

        return formatted
