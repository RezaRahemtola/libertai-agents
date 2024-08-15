from enum import Enum

from pydantic import BaseModel


class MessageRoleEnum(str, Enum):
    system = 'system'
    user = 'user'
    assistant = 'assistant'


class Message(BaseModel):
    role: MessageRoleEnum
    content: str


class LlamaCppParams(BaseModel):
    prompt: str
