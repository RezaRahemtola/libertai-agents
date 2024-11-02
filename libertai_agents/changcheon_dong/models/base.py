import logging
from abc import ABC, abstractmethod
from typing import Literal

from libertai_agents.interfaces.messages import Message, ToolCallFunction, MessageRoleEnum

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
    include_system_message: bool

    def __init__(self, model_id: ModelId, vm_url: str, context_length: int, include_system_message: bool = True):
        """
        Creates a new instance of a model

        :param model_id: HuggingFace ID of the model
        :param vm_url: URL of the completion endpoint
        :param context_length: Number of tokens allowed
        :param include_system_message: Define if a system message is supported for this model
        """
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model_id = model_id
        self.vm_url = vm_url
        self.context_length = context_length
        self.include_system_message = include_system_message

    def __count_tokens(self, content: str) -> int:
        """
        Count the number of tokens used in a string prompt

        :param content: Prompt to count
        :return: Number of token used
        """
        tokens = self.tokenizer.tokenize(content)
        return len(tokens)

    def generate_prompt(self, messages: list[Message], tools: list, system_prompt: str | None = None) -> str:
        """
        Generate the whole chat prompt

        :param messages: Messages conversation history
        :param system_prompt: Prompt to include in the beginning
        :param tools: Available tools
        :return: Prompt string
        """
        system_messages = [Message(role=MessageRoleEnum.system,
                                   content=system_prompt)] if self.include_system_message and system_prompt is not None else []
        raw_messages = list(map(lambda x: x.dict(), messages))

        for i in range(len(raw_messages)):
            included_messages: list = system_messages + raw_messages[i:]
            prompt = self.tokenizer.apply_chat_template(conversation=included_messages, tools=tools,
                                                        tokenize=False,
                                                        add_generation_prompt=True)
            if not isinstance(prompt, str):
                raise TypeError("Generated prompt isn't a string")
            if self.__count_tokens(prompt) <= self.context_length:
                return prompt
        raise ValueError(f"Can't fit messages into the available context length ({self.context_length} tokens)")

    def generate_tool_call_id(self) -> str | None:
        """
        Generate a random ID for a tool call
        
        :return: A string, or None if this model doesn't require a tool call ID
        """
        return None

    @staticmethod
    @abstractmethod
    def extract_tool_calls_from_response(response: str) -> list[ToolCallFunction]:
        """
        Extract tool calls (if any) from a given model response

        :param response: Model response to parse
        :return: List of found tool calls
        """
        pass