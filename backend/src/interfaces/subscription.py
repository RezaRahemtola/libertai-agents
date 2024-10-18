# TODO: make a shared package for these types
from enum import Enum

from pydantic import BaseModel


class SubscriptionChain(str, Enum):
    base = "base"


class SubscriptionAccount(BaseModel):
    address: str
    chain: SubscriptionChain

    class Config:
        schema_extra = {"example": {"address": "0x0000000000000000000000000000000000000000", "chain": "base"}}
