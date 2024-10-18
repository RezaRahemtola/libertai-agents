from pydantic import BaseModel


class ModelInformation(BaseModel):
    id: str
    context_length: int
