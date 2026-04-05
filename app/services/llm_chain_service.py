"""Service for LLM chain creation and invocation."""
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)


class LLMChainService:
    """Responsible for LLM chain setup and execution."""

    def __init__(self, model: str, prompt_template: str, temperature: float = 0.7, enable_thinking: bool = True):
        self.model_name = model
        self.temperature = temperature
        self.enable_thinking = enable_thinking
        self.llm = ChatOllama(
            model=model,
            temperature=temperature,
            reasoning=True
        )
        self.prompt = ChatPromptTemplate.from_template(prompt_template)
        self.chain = self.prompt | self.llm

    def invoke(self, question: str, context: str) -> str:
        """Invoke chain with question and context."""
        logger.info(f"Invoking LLM chain with {self.model_name}... (thinking={'enabled' if self.enable_thinking else 'disabled'})")
        logger.debug(f"Context length: {len(str(context))} characters")

        response = self.chain.invoke({
            "context": context,
            "question": question
        })

        for chunk in self.chain.stream({"context": context, "question": question}):
            # Extract reasoning from additional_kwargs
            thinking = chunk.additional_kwargs.get("reasoning_content")

            if thinking:
                # Print in Gray (\033[90m)
                print(f"\033[90m{thinking}\033[0m", end="", flush=True)
            elif chunk.content:
                # Detect transition to final answer
                if not hasattr(self.llm, '_started_answer'):
                    print("\n\n--- FINAL ANSWER ---")
                    self.llm._started_answer = True
                # Print in Default color
                print(chunk.content, end="", flush=True)

        # Extract string content from AIMessage
        response_text = response.content if hasattr(response, 'content') else str(response)
        logger.info(f"LLM response generated ({len(response_text)} characters)")
        return response_text
