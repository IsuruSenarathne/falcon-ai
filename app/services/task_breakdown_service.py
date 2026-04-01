import json
import re
from typing import List, Tuple

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.dto.conversation_dto import TaskBreakdownRequest, TaskBreakdownResponse, TaskItem


class TaskBreakdownService:
    """Service for analyzing user input, classifying it, and breaking down into tasks if needed."""

    def __init__(self):
        # First chain: Understand the type of input
        classification_template = """Analyze the following user input and determine what type it is.

User Input: {statement}

Respond with EXACTLY this format (no additional text):

===TYPE_START===
[one of: question, request, statement, command, problem, suggestion, other]
===TYPE_END===

===CATEGORY_START===
[one of: information_seeking, action_needed, analysis_needed, clarification_needed, feedback, general_conversation]
===CATEGORY_END===

===INTENT_START===
[Brief one sentence describing what the user wants or is asking]
===INTENT_END===

Rules:
- Type: question (ends with ?), request (asks for action), statement (makes a claim), command (direct instruction), problem (describes an issue), suggestion (proposes idea), other
- Category: What kind of response/action is needed
- Intent: The underlying goal
"""

        self.classification_chain = (
            ChatPromptTemplate.from_template(classification_template)
            | ChatOllama(model="llama3.2:1b")
            | StrOutputParser()
        )

        # Second chain: Break down into tasks (if applicable)
        breakdown_template = """Based on the user input and its classification, break it down into tasks if appropriate.

User Input: {statement}
Input Type: {input_type}
Category: {category}
Intent: {intent}

If the input requires action, problem-solving, or planning, respond with:

===SUMMARY_START===
[Brief one-sentence summary of what needs to be done]
===SUMMARY_END===

===TASKS_START===
[
  {{"id": 1, "title": "Task title", "description": "What needs to be done", "priority": "high|medium|low", "estimated_effort": "quick|medium|complex"}},
  {{"id": 2, "title": "Task title", "description": "What needs to be done", "priority": "high|medium|low", "estimated_effort": "quick|medium|complex"}}
]
===TASKS_END===

If the input does NOT require breaking down (e.g., simple question, general conversation), respond with:

===SUMMARY_START===
[Brief summary of the input]
===SUMMARY_END===

===TASKS_START===
[]
===TASKS_END===

Rules:
- Only create tasks for action-oriented inputs (requests, problems, commands)
- Questions and statements about information do NOT need tasks
- Return valid JSON array (empty [] if no tasks needed)
- For tasks: 2-6 items, ordered by priority
"""

        self.breakdown_chain = (
            ChatPromptTemplate.from_template(breakdown_template)
            | ChatOllama(model="llama3.2:1b")
            | StrOutputParser()
        )

    def breakdown(self, req: TaskBreakdownRequest) -> TaskBreakdownResponse:
        """Analyze user input, classify it, and break into tasks if needed."""
        if not req.statement or not req.statement.strip():
            raise ValueError("Statement cannot be empty")

        try:
            # Step 1: Classify the input
            classification = self._classify_input(req.statement)
            input_type = classification.get("type", "other")
            category = classification.get("category", "general_conversation")
            intent = classification.get("intent", "")

            # Step 2: Break down into tasks if applicable
            summary, tasks = self._breakdown_if_needed(
                req.statement,
                input_type,
                category,
                intent
            )

            return TaskBreakdownResponse(
                statement=req.statement,
                summary=summary,
                tasks=tasks,
                total_tasks=len(tasks),
                status="success"
            )

        except ValueError as e:
            return TaskBreakdownResponse(
                statement=req.statement,
                summary="",
                tasks=[],
                total_tasks=0,
                status="error",
                error=str(e)
            )
        except Exception as e:
            return TaskBreakdownResponse(
                statement=req.statement,
                summary="",
                tasks=[],
                total_tasks=0,
                status="error",
                error=f"Failed to analyze request: {str(e)}"
            )

    def _classify_input(self, statement: str) -> dict:
        """Classify the type of user input."""
        try:
            raw_response = self.classification_chain.invoke({"statement": statement})

            # Parse classification
            classification = {
                "type": self._extract_section(raw_response, "TYPE"),
                "category": self._extract_section(raw_response, "CATEGORY"),
                "intent": self._extract_section(raw_response, "INTENT"),
            }

            return classification
        except Exception as e:
            print(f"Classification failed: {e}")
            return {
                "type": "other",
                "category": "general_conversation",
                "intent": statement[:100]
            }

    def _breakdown_if_needed(self, statement: str, input_type: str, category: str, intent: str) -> Tuple[str, list]:
        """Break down into tasks only if input requires action."""
        try:
            raw_response = self.breakdown_chain.invoke({
                "statement": statement,
                "input_type": input_type,
                "category": category,
                "intent": intent,
            })

            summary = self._extract_section(raw_response, "SUMMARY")
            tasks = self._parse_tasks(self._extract_section(raw_response, "TASKS"))

            return summary, tasks
        except Exception as e:
            print(f"Breakdown failed: {e}")
            return "", []

    def _extract_section(self, response: str, section: str) -> str:
        """Extract a section from the response."""
        start_marker = f"==={section}_START==="
        end_marker = f"==={section}_END==="

        if start_marker in response and end_marker in response:
            start = response.find(start_marker) + len(start_marker)
            end = response.find(end_marker)
            return response[start:end].strip()

        return ""

    def _parse_tasks(self, tasks_json_str: str) -> List[TaskItem]:
        """Parse tasks from JSON string."""
        tasks = []

        if not tasks_json_str or tasks_json_str == "[]":
            return tasks

        try:
            tasks_data = json.loads(tasks_json_str)

            for task_data in tasks_data:
                if isinstance(task_data, dict):
                    task = TaskItem(
                        id=int(task_data.get("id", 0)),
                        title=str(task_data.get("title", "")).strip(),
                        description=str(task_data.get("description", "")).strip(),
                        priority=str(task_data.get("priority", "medium")).lower(),
                        estimated_effort=str(task_data.get("estimated_effort", "medium")).lower()
                    )
                    if task.title and task.description:
                        tasks.append(task)
        except json.JSONDecodeError as e:
            # Try to extract JSON from malformed response
            json_match = re.search(r'\[.*\]', tasks_json_str, re.DOTALL)
            if json_match:
                try:
                    tasks_data = json.loads(json_match.group())
                    for task_data in tasks_data:
                        if isinstance(task_data, dict):
                            task = TaskItem(
                                id=int(task_data.get("id", 0)),
                                title=str(task_data.get("title", "")).strip(),
                                description=str(task_data.get("description", "")).strip(),
                                priority=str(task_data.get("priority", "medium")).lower(),
                                estimated_effort=str(task_data.get("estimated_effort", "medium")).lower()
                            )
                            if task.title and task.description:
                                tasks.append(task)
                except json.JSONDecodeError:
                    pass

        return tasks
