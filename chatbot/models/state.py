"""
State models for the chatbot.
"""
from typing import Sequence, TypedDict, Annotated, List, Optional, Any

from langchain_core.messages import BaseMessage
from langchain_core.documents import Document
from langgraph.graph.message import add_messages


def sum_values(value: int, current_value: int) -> int:
    return current_value + value


def extend_array(value: List[int], current_value: List[int]) -> List[int]:
    return current_value + value


def conditional_string_update(current_value: str, value: Optional[str] = None) -> str:
    if current_value:
        return current_value
    return value


class ChatbotState(TypedDict):
    """State schema for the chatbot."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    system_prompt: str
    token_count: Annotated[int, sum_values]
    token_array: Annotated[List[int], extend_array]
    retrieved_context: Optional[List[Document]]
    is_new_context: bool
    is_first_message: bool
    original_query: Annotated[str, conditional_string_update]
