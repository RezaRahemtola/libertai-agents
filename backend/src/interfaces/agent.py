from pydantic import BaseModel, validator

from src.config import config
from src.interfaces.subscription import SubscriptionAccount


class DeleteAgentBody(BaseModel):
    subscription_id: str
    password: str

    # noinspection PyMethodParameters
    @validator("password")
    def format_address(cls, password: str):
        if password != config.SUBSCRIPTION_BACKEND_PASSWORD:
            raise ValueError(
                "Invalid password, you are not authorized to call this route"
            )


class SetupAgentBody(DeleteAgentBody):
    account: SubscriptionAccount


class UpdateAgentResponse(BaseModel):
    vm_hash: str


class Agent(BaseModel):
    id: str
    subscription_id: str
    vm_hash: str | None
    encrypted_secret: str
    last_update: int
    tags: list[str]


class FetchedAgent(Agent):
    post_hash: str
