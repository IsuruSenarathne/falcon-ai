"""Service for LLM chain creation and invocation."""
import logging
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from typing import Any

logger = logging.getLogger(__name__)


class LLMChainService:
    """Responsible for LLM chain setup and execution."""

    def __init__(self, model: str, prompt_template: str, temperature: float = 0.7):
        self.model_name = model
        self.temperature = temperature
        self.llm = OllamaLLM(model=model, temperature=temperature)
        self.prompt = ChatPromptTemplate.from_template(prompt_template)
        self.chain = self.prompt | self.llm

    def invoke(self, question: str, context: str) -> str:
        """Invoke chain with question and context."""
        logger.info(f"  → Invoking LLM chain with {self.model_name}...")
        logger.debug(f"    Context length: {len(str(context))} characters")

        response = self.chain.invoke({
            "context": context,
            "question": question
        })

        logger.info(f"  ✓ LLM response generated ({len(str(response))} characters)")
        return response
