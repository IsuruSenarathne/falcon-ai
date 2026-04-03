from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.constants.models import LLM_MAIN_MODEL, EMBEDDING_MODEL
from app.dto.conversation_dto import QueryRequest, QueryResponse
from app.models.conversation import MessageStatus
from app.repositories.knowledge_repository import KnowledgeRepository
from app.services.conversation_service import ConversationService
from app.services.search_service import SearchService
from app.services.planning_service import PlanningService


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
            docs = self.vectorstore.similarity_search(question, k=5)
            if docs:
                context_parts.append("\n".join([doc.page_content for doc in docs]))
            print(f"        ✓ Datasource: {len(docs)} documents")

        if context_type == "default":
            print(f"      → Using LLM knowledge (no additional context)")

        if context_type == "web_search":
            try:
                print(f"      → Running web search...")
                fetched_content = self.search_service.search(question)
                if fetched_content:
                    # RAG on web search results: embed and retrieve only relevant chunks
                    print(f"      → Creating vectorstore from {len(fetched_content)} sources...")
                    web_texts = [f"[{item['title']}]\n{item['content']}" for item in fetched_content]
                    web_vectorstore = Chroma.from_texts(
                        texts=web_texts,
                        embedding=OllamaEmbeddings(model=EMBEDDING_MODEL)
                    )

                    # Retrieve most relevant chunks from web search results
                    web_docs = web_vectorstore.similarity_search(question, k=5)
                    if web_docs:
                        context_parts.append("\n".join([doc.page_content for doc in web_docs]))
                        print(f"        ✓ Web search RAG: {len(web_docs)} relevant chunks retrieved from {len(fetched_content)} sources")
                    else:
                        print(f"        ⚠ No relevant chunks found in web search results")
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

    def _execute_plan_step(self, step_question: str, context_type: str) -> str:
        """Execute a single step of a plan by retrieving context and getting LLM response."""
        print(f"      → Executing: {step_question[:50]}...")
        context = self._retrieve_context(step_question, context_type)
        step_response = self._invoke_llm(step_question, context, None)
        answer, _ = self._parse_response(step_response)
        print(f"        ✓ Step answer ({len(answer)} chars)")
        return answer

    def _synthesize_plan(self, original_question: str, plan_answers: list, synthesis_instruction: str, conversation_history: list = None) -> str:
        """Synthesize answers from plan steps into final response."""
        print(f"    → Synthesizing plan answers...")

        # Format plan answers for context
        synthesis_context = "Step Results:\n"
        for i, answer in enumerate(plan_answers, 1):
            synthesis_context += f"\nStep {i} Answer:\n{answer}\n"

        synthesis_context += f"\nSynthesis Instruction: {synthesis_instruction}"

        # Create synthesis prompt
        synthesis_template = """Based on the following step results and synthesis instruction, provide a comprehensive final answer to the original question:

Original Question: {question}

{step_results}

{conversation_context}

Respond with your answer in HTML, then a blank line, then the line "---REASONING---", then your reasoning.

Example format:
<p>Your synthesized answer here</p>

---REASONING---
Your reasoning here"""

        conversation_context = self._format_conversation_context(conversation_history)

        chain = (
            ChatPromptTemplate.from_template(synthesis_template)
            | self.llm
            | StrOutputParser()
        )

        raw_response = chain.invoke({
            "question": original_question,
            "step_results": synthesis_context,
            "conversation_context": conversation_context
        })

        print(f"      ✓ Synthesis complete")
        return raw_response

    def query(self, req: QueryRequest) -> QueryResponse:
        if not req.question or not req.question.strip():
            raise ValueError("Question cannot be empty")

        print(f"\n{'='*60}")
        print(f"📚 RAGService.query() starting for: {req.question[:50]}...")
        print(f"{'='*60}")

        try:
            # 0. Analyze question with planning
            print(f"  → Analyzing question with planning...")
            plan = self.planning_service.analyze(req.question)
            print(f"    ✓ Plan created (complex: {plan.is_complex}, context: {plan.context_type})")

            # 1. Fetch conversation history if this is a follow-up question
            conversation_history = None
            if req.session_id:
                print(f"  → Fetching conversation history...")
                conversation_history = ConversationService.get_conversation_history(req.session_id)
                if conversation_history:
                    print(f"    ✓ Found {len(conversation_history)} previous messages")

            # 2. Determine context type (override: request > plan > default)
            if req.context_type != "default":
                context_type = req.context_type
                print(f"  → Using context type from request: {context_type}")
            else:
                context_type = plan.context_type if plan.needs_context else "default"
                print(f"  → Using context type from plan: {context_type}")

            # 3. Execute plan or simple query
            raw_response = ""
            if plan.is_complex:
                print(f"  → Executing complex query plan ({len(plan.steps)} steps)...")

                # Execute each step
                step_answers = []
                for step in plan.steps:
                    step_answer = self._execute_plan_step(step.question, context_type)
                    step_answers.append(step_answer)

                # Synthesize answers
                raw_response = self._synthesize_plan(
                    req.question,
                    step_answers,
                    plan.synthesis_instruction,
                    conversation_history
                )
            else:
                print(f"  → Executing simple query (no planning needed)...")

                # 3a. Retrieve context
                context = ""
                if context_type != "default":
                    print(f"    → Retrieving context (type: {context_type})...")
                    context = self._retrieve_context(req.question, context_type)
                    print(f"    ✓ Context retrieved ({len(context)} chars)")
                else:
                    print(f"    → Using LLM knowledge (no additional context)")

                # 3b. Make SINGLE LLM call with question + context
                print(f"    → Invoking LLM...")
                raw_response = self._invoke_llm(req.question, context, conversation_history)

            print(f"  ✓ LLM response received")

            # 4. Parse response
            answer, reasoning = self._parse_response(raw_response)
            print(f"    ✓ Parsed answer ({len(answer)} chars)")

            # 5. Process success and save to database
            result = self._process_success(req, answer, reasoning)
            print(f"✓ RAGService completed")
            print(f"{'='*60}\n")

            return result

        except ValueError:
            raise
        except Exception as e:
            print(f"Error: {str(e)}")
            print(f"{'='*60}\n")
            return self._process_error(req, str(e))
