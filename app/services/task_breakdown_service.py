import json
import re
import time
from typing import List

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.dto.conversation_dto import TaskBreakdownRequest, TaskBreakdownResponse, TaskItem


class TaskBreakdownService:
    """Service for analyzing user input, classifying it, and breaking down into tasks if needed."""

    def __init__(self):
        # Combined chain: Classify AND break down in ONE call
        combined_template = """Analyze the following user input and determine what type it is, then break it down into tasks if needed.

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

===SUMMARY_START===
[Brief one-sentence summary of what needs to be done or answered]
===SUMMARY_END===

===TASKS_START===
[
  {{"id": 1, "title": "Task title", "description": "What needs to be done", "priority": "high|medium|low", "estimated_effort": "quick|medium|complex"}},
  {{"id": 2, "title": "Task title", "description": "What needs to be done", "priority": "high|medium|low", "estimated_effort": "quick|medium|complex"}}
]
===TASKS_END===

Rules:
- Type: question (ends with ?), request (asks for action), statement (makes a claim), command (direct instruction), problem (describes an issue), suggestion (proposes idea), other
- Category: What kind of response/action is needed
- Intent: The underlying goal
- Summary: Brief description
- Tasks: Only for action-oriented inputs (requests, problems, commands). Return empty [] for questions and information-seeking. Return 2-6 tasks when needed.
"""

        self.combined_chain = (
            ChatPromptTemplate.from_template(combined_template)
            | ChatOllama(model="qwen2.5:1.5b")
            | StrOutputParser()
        )

    def breakdown(self, req: TaskBreakdownRequest) -> TaskBreakdownResponse:
        """Analyze user input, classify it, and break into tasks in ONE LLM call."""
        if not req.statement or not req.statement.strip():
            raise ValueError("Statement cannot be empty")

        start_time = time.time()

        try:
            # Single LLM call for classification + breakdown
            llm_start = time.time()
            raw_response = self.combined_chain.invoke({"statement": req.statement})
            llm_time = time.time() - llm_start
            print(f"⏱ Classification + Breakdown (1 LLM call): {llm_time:.2f}s")

            # Parse all sections
            intent = self._extract_section(raw_response, "INTENT")
            summary = self._extract_section(raw_response, "SUMMARY")
            tasks = self._parse_tasks(self._extract_section(raw_response, "TASKS"))

            total_time = time.time() - start_time
            print(f"✅ Task Breakdown Total: {total_time:.2f}s\n")

            return TaskBreakdownResponse(
                statement=req.statement,
                summary=summary or intent,
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
