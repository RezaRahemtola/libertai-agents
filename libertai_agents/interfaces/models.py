from pydantic import BaseModel

from libertai_agents.models.base import ModelId


class ModelInformation(BaseModel):
    id: ModelId
    context_length: int
