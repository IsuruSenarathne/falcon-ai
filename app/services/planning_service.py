import json
import time
from dataclasses import dataclass
from typing import List

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.constants.models import LLM_MAIN_MODEL


@dataclass
class PlanStep:
    """Represents a single step in a plan"""
    step_number: int
    question: str
    description: str


@dataclass
class Plan:
    """Plan for answering a complex question"""
    is_complex: bool  # True if multi-step planning needed
    original_question: str
    steps: List[PlanStep]
    synthesis_instruction: str
    needs_context: bool = True  # Whether context is needed for this query
    context_type: str = "default"  # Which context type: "datasource", "web_search", or "default" (LLM knowledge)


class PlanningService:
    """Service for breaking down complex questions into steps and creating plans."""

    def __init__(self):
        # Complexity detection template
        complexity_template = """Analyze this question and determine:
1. If it's complex (requires multiple steps to answer)
2. If it needs external context/knowledge to answer
3. What type of context: "datasource" (internal knowledge base), "web_search" (web), or "default" (LLM's own knowledge)

Question: {question}

A complex question typically:
- Asks for multiple items/comparisons (e.g., "find me...", "compare...", "list...")
- Has multiple conditions/criteria (e.g., "X AND Y AND Z")
- Requires combining information from different domains
- Asks for filtering/ranking across multiple dimensions

Context needed for:
- "datasource": Questions about specific topics in internal knowledge base (courses, advisors, departments, etc.)
- "web_search": Questions requiring current/real-world information (latest prices, current events, recent data)
- "default": Greetings, general knowledge, or conversational queries that need no external context

Return ONLY valid JSON with no other text:
{{
    "is_complex": true|false,
    "needs_context": true|false,
    "context_type": "datasource"|"web_search"|"default",
    "reason": "brief explanation"
}}"""

        self.complexity_chain = (
            ChatPromptTemplate.from_template(complexity_template)
            | ChatOllama(model=LLM_MAIN_MODEL)
            | JsonOutputParser()
        )

        # Planning template for complex questions
        planning_template = """Break down this complex question into 2-4 logical steps that can be answered independently, then synthesized.

Question: {question}

Return ONLY valid JSON with no other text:
{{
    "steps": [
        {{
            "step_number": 1,
            "question": "sub-question 1 to answer",
            "description": "what information this step gathers"
        }},
        {{
            "step_number": 2,
            "question": "sub-question 2 to answer",
            "description": "what information this step gathers"
        }}
    ],
    "synthesis_instruction": "how to combine answers from all steps into final answer"
}}"""

        self.planning_chain = (
            ChatPromptTemplate.from_template(planning_template)
            | ChatOllama(model=LLM_MAIN_MODEL)
            | JsonOutputParser()
        )

    def analyze(self, question: str) -> Plan:
        """Analyze question and create plan if needed."""
        start = time.time()
        print(f"\n  🧠 PlanningService.analyze() starting...")

        try:
            # Step 1: Check complexity
            print(f"    → Detecting complexity...")
            complexity_start = time.time()
            complexity_result = self.complexity_chain.invoke({"question": question})
            complexity_time = time.time() - complexity_start
            print(f"      ✓ Complexity detection: {complexity_time:.2f}s")

            is_complex = complexity_result.get("is_complex", False)
            needs_context = complexity_result.get("needs_context", True)
            context_type = complexity_result.get("context_type", "default")
            reason = complexity_result.get("reason", "")
            print(f"      Is complex: {is_complex}")
            print(f"      Needs context: {needs_context} (type: {context_type})")
            print(f"      Reason: {reason}")

            # Step 2: If complex, create plan
            if is_complex:
                print(f"    → Creating execution plan...")
                planning_start = time.time()
                planning_result = self.planning_chain.invoke({"question": question})
                planning_time = time.time() - planning_start
                print(f"      ✓ Plan created: {planning_time:.2f}s")

                steps_data = planning_result.get("steps", [])
                steps = [
                    PlanStep(
                        step_number=s.get("step_number"),
                        question=s.get("question"),
                        description=s.get("description")
                    )
                    for s in steps_data
                ]
                synthesis = planning_result.get("synthesis_instruction", "")

                print(f"      Plan has {len(steps)} steps:")
                for step in steps:
                    print(f"        Step {step.step_number}: {step.question[:50]}...")

                plan = Plan(
                    is_complex=True,
                    original_question=question,
                    steps=steps,
                    synthesis_instruction=synthesis,
                    needs_context=needs_context,
                    context_type=context_type
                )
            else:
                # Simple question, no steps needed
                plan = Plan(
                    is_complex=False,
                    original_question=question,
                    steps=[],
                    synthesis_instruction="",
                    needs_context=needs_context,
                    context_type=context_type
                )

            total_time = time.time() - start
            print(f"  ✅ Planning completed in {total_time:.2f}s\n")
            print(f"plan: {plan}")
            return plan

        except Exception as e:
            print(f"  ❌ Planning failed: {str(e)}")
            # Return non-complex plan as fallback with no context
            return Plan(
                is_complex=False,
                original_question=question,
                steps=[],
                synthesis_instruction="",
                needs_context=False,
                context_type="default"
            )
