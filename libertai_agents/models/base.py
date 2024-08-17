import logging
from abc import ABC, abstractmethod

from libertai_agents.interfaces import Message, ToolCallFunction, MessageRoleEnum

# Disables the error about models not available
logging.getLogger("transformers").disabled = True


class Model(ABC):
    from transformers import PreTrainedTokenizerFast

    tokenizer: PreTrainedTokenizerFast
    vm_url: str
    system_message: bool

    def __init__(self, model_id: str, vm_url: str, system_message: bool = True):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.vm_url = vm_url
        self.system_message = system_message

    def generate_prompt(self, messages: list[Message], system_prompt: str, tools: list) -> str:
        if self.system_message:
            messages.insert(0, Message(role=MessageRoleEnum.system, content=system_prompt))
        raw_messages = list(map(lambda x: x.model_dump(), messages))

        return self.tokenizer.apply_chat_template(conversation=raw_messages, tools=tools, tokenize=False,
                                                  add_generation_prompt=True)

    def generate_tool_call_id(self) -> str | None:
        return None

    @staticmethod
    @abstractmethod
    def extract_tool_calls_from_response(response: str) -> list[ToolCallFunction]:
        pass
