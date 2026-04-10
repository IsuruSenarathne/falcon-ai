import json
from dataclasses import dataclass
from typing import List

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.constants.models import LLM_MAIN_MODEL


@dataclass
class ThinkingResult:
    """Result of thinking about a question"""
    what_user_asking: str  # what the user actually wants
    plan: str  # how to answer it


@dataclass
class ContextSources:
    """Context sources to use for answering the question"""
    sources: List[str]  # list of: "llm_knowledge", "web_search", "database"


class ThinkingService:
    """Service that understands what user is asking and plans the answer."""

    def __init__(self):
        template = """Analyze this question and respond as JSON with no other text.

Question: {question}

Return ONLY valid JSON with this structure:
{{
    "what_user_asking": "clear description of what user is asking for",
    "plan": "step-by-step plan for how to answer this"
}}"""

        self.chain = (
            ChatPromptTemplate.from_template(template)
            | ChatOllama(model=LLM_MAIN_MODEL)
            | JsonOutputParser()
        )

        # Context source decision template
        context_template = """Decide which context sources would best answer this question.

Question: {question}

Available sources:
- "llm_knowledge": LLM's built-in knowledge (definitions, explanations, general facts)
- "web_search": Real-time web search using Brave (latest news, current events, recent info)
- "database": Internal knowledge base (courses, advisors, departments, policies)

Return ONLY valid JSON with no other text:
{{
    "sources": ["source1", "source2"] // ordered by relevance, can be 1-3 sources
}}

Rules:
- Use "llm_knowledge" for definitions, explanations, general concepts
- Use "web_search" for current events, latest news, real-time information, trending topics
- Use "database" for internal company/institution information
- Use combinations when multiple sources would help (e.g., ["database", "llm_knowledge"])
"""

        self.context_chain = (
            ChatPromptTemplate.from_template(context_template)
            | ChatOllama(model=LLM_MAIN_MODEL)
            | JsonOutputParser()
        )

    def think(self, question: str) -> ThinkingResult:
        """Understand question and create answer plan."""
        try:
            response = self.chain.invoke({"question": question})

            return ThinkingResult(
                what_user_asking=response.get("what_user_asking", ""),
                plan=response.get("plan", "")
            )
        except Exception as e:
            print(f"Thinking failed: {e}")
            return ThinkingResult(
                what_user_asking=question,
                plan="Answer the question directly using available knowledge"
            )

    def decide_context_sources(self, what_user_asking: str) -> ContextSources:
        """Decide which context sources to use for answering the question."""
        try:
            response = self.context_chain.invoke({"question": what_user_asking})
            sources = response.get("sources", ["llm_knowledge"])

            # Validate sources
            valid_sources = ["llm_knowledge", "web_search", "database"]
            sources = [s for s in sources if s in valid_sources]

            if not sources:
                sources = ["llm_knowledge"]
            
            print(f"📚 Context sources decided: {sources}")
            return ContextSources(sources=sources)
        except Exception as e:
            print(f"Context decision failed: {e}")
            return ContextSources(sources=["llm_knowledge", "database"])

