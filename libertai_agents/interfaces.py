from enum import Enum
from typing import Optional

from pydantic import BaseModel


class MessageRoleEnum(str, Enum):
    system = 'system'
    user = 'user'
    assistant = 'assistant'
    tool = 'tool'


class ToolCallFunction(BaseModel):
    name: str
    arguments: dict


class MessageToolCall(BaseModel):
    type: str
    id: Optional[str] = None
    function: ToolCallFunction


class Message(BaseModel):
    role: MessageRoleEnum
    name: Optional[str] = None
    content: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[list[MessageToolCall]] = None


class LlamaCppParams(BaseModel):
    prompt: str
