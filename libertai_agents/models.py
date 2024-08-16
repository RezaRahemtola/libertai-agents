import json
import re

from transformers import AutoTokenizer, PreTrainedTokenizerFast

from libertai_agents.interfaces import Message, MessageRoleEnum, ToolCallFunction


class Model:
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

    @staticmethod
    def extract_tool_calls_from_response(response: str) -> list[ToolCallFunction]:
        tool_calls = re.findall("^<tool_call>\s*(.*)\s*</tool_call>$", response)
        return [ToolCallFunction(**json.loads(call)) for call in tool_calls]


Hermes2Pro = Model(model_id="NousResearch/Hermes-2-Pro-Llama-3-8B",
                   vm_url='https://curated.aleph.cloud/vm/84df52ac4466d121ef3bb409bb14f315de7be4ce600e8948d71df6485aa5bcc3/completion')
