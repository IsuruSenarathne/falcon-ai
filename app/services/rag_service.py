import time

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.constants.models import LLM_MAIN_MODEL, EMBEDDING_MODEL
from app.dto.conversation_dto import QueryRequest, QueryResponse
from app.models.conversation import MessageStatus
from app.repositories.knowledge_repository import KnowledgeRepository
from app.services.conversation_service import ConversationService
from app.services.planning_service import PlanningService
from app.services.search_service import SearchService


class RAGService:

    def __init__(self, task_breakdown_service=None):
        self.task_breakdown_service = task_breakdown_service
        self.search_service = SearchService()
        self.planning_service = PlanningService()
        
        documents = KnowledgeRepository.load_documents()
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        self.vectorstore = Chroma.from_texts(texts=documents, embedding=embeddings)
        print(f"✓ Knowledge base loaded: {len(documents)} documents from database")

        # Template for knowledge base context
        self.kb_template = """Answer the question based ONLY on the following context:
{context}

{conversation_context}

Question: {question}

Instructions:
- If the question asks for a list, comparison, or "find me..." → provide a COMPREHENSIVE list with all relevant options from the context
- If listing items → use <ul><li> format with details for each item
- Be thorough and include all relevant information available in the context

Respond with your answer in HTML, then a blank line, then the line "---REASONING---", then your reasoning.

Do NOT include brackets, labels, or placeholder text. Just the actual content.

Example format:
<p>Your answer here</p>

---REASONING---
Your reasoning here

Rules:
- Use valid HTML tags for formatting
- Explain how you derived the answer from the context
- Do NOT include diagrams, charts, or images
- Inner HTML only (no <html>, <head>, <body>, <style>, <script> tags)
"""

        # Template for general knowledge (no context)
        self.general_template = """Answer the following question using your knowledge:

{conversation_context}

Question: {question}

Instructions:
- If the question asks for a list, comparison, or "find me..." → provide a COMPREHENSIVE list with all relevant options you know about
- If listing items → use <ul><li> format with details for each item
- Be thorough and include all relevant information you have

Respond with your answer in HTML, then a blank line, then the line "---REASONING---", then your reasoning.

Do NOT include brackets, labels, or placeholder text. Just the actual content.

Example format:
<p>Your answer here</p>

---REASONING---
Your reasoning here

Rules:
- Use valid HTML tags for formatting
- Explain your reasoning
- Do NOT include diagrams, charts, or images
- Inner HTML only (no <html>, <head>, <body>, <style>, <script> tags)
"""

        self.llm = ChatOllama(model=LLM_MAIN_MODEL)

    def _parse_response(self, raw_response: str) -> tuple[str, str]:
        """Parse raw response into answer and reasoning sections."""
        answer = ""
        reasoning = ""

        # Parse by ---REASONING--- separator
        if "---REASONING---" in raw_response:
            parts = raw_response.split("---REASONING---")
            answer = parts[0].strip()
            reasoning = parts[1].strip() if len(parts) > 1 else ""
        elif "REASONING:" in raw_response:
            # Fallback for old ANSWER:/REASONING: format
            parts = raw_response.split("REASONING:")
            answer = parts[0].replace("ANSWER:", "").strip()
            reasoning = parts[1].strip() if len(parts) > 1 else ""
        else:
            # Last resort fallback
            answer = raw_response.strip()
            reasoning = ""

        return answer, reasoning

    def _format_with_tasks_and_reasoning(self, answer: str, reasoning: str, tasks_html: str = "") -> str:
        """Format answer with collapsible tasks and reasoning sections."""
        formatted = answer

        if tasks_html:
            formatted += f"""

<details>
<summary>Tasks</summary>
{tasks_html}
</details>"""

        if reasoning:
            formatted += f"""

<details>
<summary>Reasoning</summary>
{reasoning}
</details>"""

        return formatted

    def _format_with_reasoning(self, answer: str, reasoning: str) -> str:
        """Format answer with collapsible reasoning section."""
        formatted = answer
        if reasoning:
            formatted += f"""

<details>
<summary>Reasoning</summary>
{reasoning}
</details>"""
        return formatted

    def _retrieve_context(self, question: str, context_type: str = "default") -> str:
        """Retrieve context based on specified type.

        Args:
            question: The question to retrieve context for
            context_type: "default" (datasource + llm), "web_search", or "datasource"
        """
        context_parts = []

        if context_type == "datasource":
            print(f"      → Searching datasource...")
            ds_start = time.time()
            docs = self.vectorstore.similarity_search(question, k=5)
            if docs:
                context_parts.append("\n".join([doc.page_content for doc in docs]))
            ds_time = time.time() - ds_start
            print(f"        ✓ Datasource: {len(docs)} documents in {ds_time:.2f}s")

        if context_type == "default":
            print(f"      → Using LLM knowledge (no additional context)")

        if context_type == "web_search":
            try:
                print(f"      → Running web search...")
                fetched_content = self.search_service.search(question)
                if fetched_content:
                    formatted = self.search_service.format_results(fetched_content)
                    context_parts.append(formatted)
                    print(f"        ✓ Web search: {len(fetched_content)} sources retrieved")
            except Exception as e:
                print(f"        ⚠ Web search failed, falling back to LLM: {str(e)[:50]}")

        context = "\n".join(context_parts)
        return context

    def _format_conversation_context(self, conversation_history: list = None) -> str:
        """Format conversation history for LLM context."""
        if not conversation_history:
            return ""

        history_text = "Previous conversation:\n"
        for msg in conversation_history[-5:]:  # Last 5 messages for context
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            history_text += f"{role}: {content}\n"

        return history_text

    def _invoke_llm(self, question: str, context: str = "", conversation_history: list = None) -> str:
        """Invoke LLM chain with appropriate template."""
        print(f"      → Formatting conversation context...")
        conversation_context = self._format_conversation_context(conversation_history)

        # Print full context before sending to LLM
        print(f"\n{'='*60}")
        print(f"📋 FULL CONTEXT BEING SENT TO LLM:")
        print(f"{'='*60}")

        if conversation_context:
            print(f"\n🔹 CONVERSATION CONTEXT:")
            print(f"{'-'*60}")
            print(conversation_context)
            print(f"{'-'*60}\n")

        if context:
            print(f"🔹 RETRIEVED CONTEXT ({len(context)} chars):")
            print(f"{'-'*60}")
            # Print first 1000 chars to avoid too much output
            context_preview = context[:1000] + "..." if len(context) > 1000 else context
            print(context_preview)
            print(f"{'-'*60}\n")

        print(f"🔹 QUESTION:")
        print(f"{'-'*60}")
        print(question)
        print(f"{'-'*60}\n")

        print(f"🔹 TEMPLATE BEING USED:")
        if context:
            print(f"Knowledge Base Template (with context)")
        else:
            print(f"General Knowledge Template (no context)")
        print(f"{'='*60}\n")

        if context:
            print(f"      → Building KB template chain ({len(context)} chars context)...")
            chain = (
                ChatPromptTemplate.from_template(self.kb_template)
                | self.llm
                | StrOutputParser()
            )
            print(f"      → Invoking LLM with context...")
            raw_response = chain.invoke({
                "context": context,
                "conversation_context": conversation_context,
                "question": question
            })
        else:
            print(f"      → Building general knowledge template chain...")
            chain = (
                ChatPromptTemplate.from_template(self.general_template)
                | self.llm
                | StrOutputParser()
            )
            print(f"      → Invoking LLM without context...")
            raw_response = chain.invoke({
                "conversation_context": conversation_context,
                "question": question
            })

        print(f"        ✓ LLM returned response ({len(raw_response)} chars)")
        return raw_response

    def _process_success(self, req: QueryRequest, answer: str, reasoning: str) -> QueryResponse:
        """Process successful query and save to database."""
        formatted_answer = self._format_with_reasoning(answer, reasoning)

        conversation_id, bot_msg = ConversationService.save_exchange(
            question=req.question,
            answer=formatted_answer,
            status=MessageStatus.SUCCESS,
            user_id=req.user_id,
            session_id=req.session_id,
            response_time=0,
        )

        return QueryResponse(
            conversation_id=conversation_id,
            question=req.question,
            answer=formatted_answer,
            status="success",
            response_time=0,
            created_at=bot_msg.created_at.isoformat(),
        )

    def _execute_step(self, step_question: str, context_type: str, conversation_history: list = None) -> str:
        """Execute a single planning step and return the answer."""
        try:
            print(f"      → Executing: {step_question[:50]}...")

            # Retrieve context for this step
            context = self._retrieve_context(step_question, context_type)

            # Invoke LLM for this step (history already filtered in query() method)
            raw_response = self._invoke_llm(step_question, context, conversation_history)

            # Parse response
            answer, _ = self._parse_response(raw_response)
            print(f"        ✓ Step answer: {len(answer)} chars")

            return answer
        except Exception as e:
            print(f"        ⚠ Step failed: {str(e)}")
            return f"Could not answer: {str(e)}"

    def _synthesize_answers(self, original_question: str, step_answers: list, synthesis_instruction: str) -> tuple:
        """Combine multiple step answers into final answer."""
        print(f"  → Synthesizing {len(step_answers)} step answers...")

        # Build synthesis prompt
        synthesis_prompt = f"""You have the following information gathered from multiple searches:

{chr(10).join([f'Step {i+1} Answer:{chr(10)}{answer}{chr(10)}' for i, answer in enumerate(step_answers)])}

Original Question: {original_question}

Synthesis Instructions: {synthesis_instruction}

Now provide a comprehensive final answer that combines all this information.

Respond with your answer in HTML, then a blank line, then the line "---REASONING---", then your reasoning.

Do NOT include brackets, labels, or placeholder text. Just the actual content.

Rules:
- Use valid HTML tags for formatting
- Create a comprehensive, well-organized answer
- Use <ul><li> for lists, <table> for comparisons if helpful
- Inner HTML only (no <html>, <head>, <body>, <style>, <script> tags)
"""

        try:
            synthesis_start = time.time()
            chain = (
                ChatPromptTemplate.from_template(synthesis_prompt)
                | self.llm
                | StrOutputParser()
            )
            raw_response = chain.invoke({})
            synthesis_time = time.time() - synthesis_start
            print(f"  ⏱️  Synthesis: {synthesis_time:.2f}s")

            answer, reasoning = self._parse_response(raw_response)
            return answer, reasoning
        except Exception as e:
            print(f"  ⚠ Synthesis failed: {str(e)}")
            # Fallback: concatenate answers
            combined = "<br>".join([f"<h3>Part {i+1}</h3>{answer}" for i, answer in enumerate(step_answers)])
            return combined, "Combined multiple step answers"

    def _process_error(self, req: QueryRequest, error: str) -> QueryResponse:
        """Process failed query and save error to database."""
        conversation_id, _ = ConversationService.save_exchange(
            question=req.question,
            answer=None,
            status=MessageStatus.ERROR,
            error=error,
            user_id=req.user_id,
            session_id=req.session_id,
            response_time=0,
        )

        return QueryResponse(
            conversation_id=conversation_id,
            question=req.question,
            answer=None,
            status="error",
            response_time=0,
            created_at="",
            error=error,
        )

    def query(self, req: QueryRequest) -> QueryResponse:
        if not req.question or not req.question.strip():
            raise ValueError("Question cannot be empty")

        start_time = time.time()
        print(f"\n{'='*60}")
        print(f"📚 RAGService.query() starting for: {req.question[:50]}...")
        print(f"{'='*60}")

        try:
            # 1. Fetch conversation history if this is a follow-up question
            conversation_history = None
            if req.session_id:
                hist_start = time.time()
                print(f"  → Fetching conversation history...")
                from app.config.database import SessionLocal
                from app.repositories.conversation_repository import ConversationRepository
                db = SessionLocal()
                try:
                    conv = ConversationRepository.find_by_id(db, req.session_id)
                    if conv:
                        conversation_history = [
                            {
                                "role": msg.role.value,
                                "content": msg.content
                            }
                            for msg in sorted(conv.messages, key=lambda m: m.created_at)
                        ]
                        print(f"    ✓ Found {len(conversation_history)} previous messages")
                finally:
                    db.close()
                hist_time = time.time() - hist_start
                print(f"  ⏱️  History fetch: {hist_time:.2f}s")

            # 2. Check if question is complex and needs planning
            plan_start = time.time()
            print(f"  → Analyzing question complexity...")
            plan = self.planning_service.analyze(req.question)
            plan_time = time.time() - plan_start
            print(f"  ⏱️  Planning phase: {plan_time:.2f}s")

            # Override context_type if explicitly specified in request (not default)
            if req.context_type != "default":
                print(f"  → Using explicit context_type from request: {req.context_type}")
                plan.context_type = req.context_type
                plan.needs_context = True  # If user explicitly asks for a context type, they need it

            # 3. Execute based on complexity
            if plan.is_complex:
                print(f"  📋 MULTI-STEP EXECUTION ({len(plan.steps)} steps):")
                step_answers = []

                for step in plan.steps:
                    print(f"\n    [Step {step.step_number}] {step.description}")
                    # Use plan's context_type decision for steps
                    # Only pass conversation history if context is needed
                    history_for_step = conversation_history if plan.needs_context else None
                    answer = self._execute_step(step.question, plan.context_type if plan.needs_context else "default", history_for_step)
                    step_answers.append(answer)

                # Synthesize all answers
                print(f"\n  → Synthesizing final answer from {len(step_answers)} steps...")
                answer, reasoning = self._synthesize_answers(
                    req.question,
                    step_answers,
                    plan.synthesis_instruction
                )
            else:
                print(f"  ℹ️  Single-step execution")

                # Standard single-step flow
                # Only retrieve context if planning decided it's needed
                context = ""
                if plan.needs_context:
                    context_start = time.time()
                    # Use context_type from planning, not from request
                    print(f"  → Retrieving context (type: {plan.context_type})...")
                    context = self._retrieve_context(req.question, plan.context_type)
                    context_time = time.time() - context_start
                    print(f"  ⏱️  Context retrieval: {context_time:.2f}s ({len(context)} chars)")
                else:
                    print(f"  → Skipping context retrieval (planning determined context not needed)")

                # Invoke LLM with conversation history
                # Only pass conversation history if context is needed (to avoid biasing LLM)
                llm_start = time.time()
                print(f"  → Invoking LLM...")
                history_for_llm = conversation_history if plan.needs_context else None
                raw_response = self._invoke_llm(req.question, context, history_for_llm)
                llm_time = time.time() - llm_start
                print(f"  ⏱️  LLM response: {llm_time:.2f}s")

                # Parse response
                answer, reasoning = self._parse_response(raw_response)
                print(f"    ✓ Parsed answer ({len(answer)} chars) + reasoning ({len(reasoning)} chars)")

            # 4. Process success and save to database
            db_start = time.time()
            result = self._process_success(req, answer, reasoning)
            db_time = time.time() - db_start
            print(f"Database save: {db_time:.2f}s")

            total_time = time.time() - start_time
            print(f"RAGService completed in {total_time:.2f}s")
            print(f"{'='*60}\n")

            return result

        except ValueError:
            raise
        except Exception as e:
            print(f"Error: {str(e)}")
            print(f"{'='*60}\n")
            return self._process_error(req, str(e))
