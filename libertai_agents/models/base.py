from abc import ABC, abstractmethod

from transformers import PreTrainedTokenizerFast, AutoTokenizer

from libertai_agents.interfaces import Message, MessageRoleEnum, ToolCallFunction


class Model(ABC):
    tokenizer: PreTrainedTokenizerFast
    vm_url: str

    def __init__(self, model_id: str, vm_url: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.vm_url = vm_url

    def generate_prompt(self, messages: list[Message], system_prompt: str, tools: list) -> str:
        messages.insert(0, Message(role=MessageRoleEnum.system, content=system_prompt))
        raw_messages = list(map(lambda x: x.model_dump(), messages))

        return self.tokenizer.apply_chat_template(conversation=raw_messages, tools=tools, tokenize=False,
                                                  add_generation_prompt=True)

    @abstractmethod
    def generate_tool_call_id(self) -> str | None:
        pass

    @staticmethod
    @abstractmethod
    def extract_tool_calls_from_response(response: str) -> list[ToolCallFunction]:
        pass
