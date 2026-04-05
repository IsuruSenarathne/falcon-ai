from dataclasses import dataclass, asdict
from typing import Optional, List


@dataclass
class QueryRequest:
    question: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    context_type: str = "default"  # "default", "web_search", "datasource"
    conversation_history: Optional[List] = None  # Previous messages in conversation

    @staticmethod
    def from_json(data: dict) -> "QueryRequest":
        return QueryRequest(
            question=data["question"],
            user_id=data.get("user_id"),
            session_id=data.get("session_id"),
            context_type=data.get("context_type", "default"),
            conversation_history=None,  # Will be set by controller
        )


@dataclass
class SearchRequest:
    question: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None

    @staticmethod
    def from_json(data: dict) -> "SearchRequest":
        return SearchRequest(
            question=data["question"],
            user_id=data.get("user_id"),
            session_id=data.get("session_id"),
        )


@dataclass
class MessageDTO:
    message_id: str
    role: str
    content: str
    created_at: Optional[str] = None
    status: Optional[str] = None
    error: Optional[str] = None
    response_time: Optional[float] = None


@dataclass
class TaskItem:
    """Represents a single task in a breakdown"""
    id: int
    title: str
    description: str
    priority: str  # "high", "medium", "low"
    estimated_effort: str  # "quick", "medium", "complex"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class QueryResponse:
    conversation_id: str
    question: str
    answer: Optional[str]
    status: str
    response_time: float
    created_at: str
    reasoning: Optional[str] = None
    tasks: Optional[List[TaskItem]] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        data = asdict(self)
        if self.tasks:
            data['tasks'] = [t.to_dict() if hasattr(t, 'to_dict') else asdict(t) for t in self.tasks]
        return data


@dataclass
class SearchResponse:
    conversation_id: str
    question: str
    answer: Optional[str]
    status: str
    response_time: float
    created_at: str
    sources: List[str]
    tasks: Optional[List[TaskItem]] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        data = asdict(self)
        if self.tasks:
            data['tasks'] = [t.to_dict() if hasattr(t, 'to_dict') else asdict(t) for t in self.tasks]
        return data


@dataclass
class ConversationDTO:
    conversation_id: str
    user_id: Optional[str]
    title: Optional[str]
    messages: List[MessageDTO]
    created_at: Optional[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ConversationsListResponse:
    conversations: List[ConversationDTO]
    total: int
    limit: int
    offset: int
    status: str = "success"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ConversationResponse:
    conversation: ConversationDTO
    status: str = "success"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MessageResponse:
    conversation_id: str
    message: MessageDTO
    status: str = "success"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MessagesResponse:
    conversation_id: str
    messages: List[MessageDTO]
    status: str = "success"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TaskBreakdownRequest:
    """Request to break down a user statement into tasks"""
    statement: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None

    @staticmethod
    def from_json(data: dict) -> "TaskBreakdownRequest":
        return TaskBreakdownRequest(
            statement=data["statement"],
            user_id=data.get("user_id"),
            session_id=data.get("session_id"),
        )


@dataclass
class TaskBreakdownResponse:
    """Response containing breakdown of tasks"""
    statement: str
    summary: str
    tasks: List[TaskItem]
    total_tasks: int
    status: str
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)
