import logging
from abc import ABC, abstractmethod
from typing import Literal

from libertai_agents.interfaces.common import Message, ToolCallFunction, MessageRoleEnum

# Disables the error about models not available
logging.getLogger("transformers").disabled = True

ModelId = Literal[
    "NousResearch/Hermes-2-Pro-Llama-3-8B",
    "NousResearch/Hermes-3-Llama-3.1-8B",
    "mistralai/Mistral-Nemo-Instruct-2407"
]


class Model(ABC):
    from transformers import PreTrainedTokenizerFast

    tokenizer: PreTrainedTokenizerFast
    model_id: ModelId
    vm_url: str
    context_length: int
    system_message: bool

    def __init__(self, model_id: ModelId, vm_url: str, context_length: int, system_message: bool = True):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model_id = model_id
        self.vm_url = vm_url
        self.context_length = context_length
        self.system_message = system_message

    def __count_tokens(self, content: str) -> int:
        tokens = self.tokenizer.tokenize(content)
        return len(tokens)

    def generate_prompt(self, messages: list[Message], system_prompt: str, tools: list) -> str:
        system_message = Message(role=MessageRoleEnum.system, content=system_prompt)
        raw_messages = list(map(lambda x: x.model_dump(), messages))

        for i in range(len(raw_messages)):
            included_messages: list = [system_message] + raw_messages[i:]
            prompt = self.tokenizer.apply_chat_template(conversation=included_messages, tools=tools,
                                                        tokenize=False,
                                                        add_generation_prompt=True)
            if not isinstance(prompt, str):
                raise TypeError("Generated prompt isn't a string")
            if self.__count_tokens(prompt) <= self.context_length:
                return prompt
        raise ValueError(f"Can't fit messages into the available context length ({self.context_length} tokens)")

    def generate_tool_call_id(self) -> str | None:
        return None

    @staticmethod
    @abstractmethod
    def extract_tool_calls_from_response(response: str) -> list[ToolCallFunction]:
        pass
