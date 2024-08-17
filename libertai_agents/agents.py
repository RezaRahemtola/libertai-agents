from typing import Callable

import aiohttp
from aiohttp import ClientSession

from libertai_agents.interfaces import Message, MessageRoleEnum, LlamaCppParams, MessageToolCall, ToolCallFunction, \
    ToolCallMessage, ToolResponseMessage
from libertai_agents.models import Model
from libertai_agents.utils import find


class ChatAgent:
    model: Model
    system_prompt: str
    tools: list[Callable]

    def __init__(self, model: Model, system_prompt: str, tools: list[Callable] | None = None):
        if tools is None:
            tools = []

        if len(set(map(lambda x: x.__name__, tools))) != len(tools):
            raise ValueError("Tool functions must have different names")
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools

    async def generate_answer(self, messages: list[Message]) -> str:
        if len(messages) == 0:
            raise ValueError("No previous message to respond to")
        if messages[-1].role not in [MessageRoleEnum.user, MessageRoleEnum.tool]:
            raise ValueError("Last message is not from the user or tool")

        prompt = self.model.generate_prompt(messages, self.system_prompt, self.tools)
        async with aiohttp.ClientSession() as session:
            response = await self.__call_model(session, prompt)

            tool_calls = self.model.extract_tool_calls_from_response(response)
            if len(tool_calls) == 0:
                return response
            tool_calls_message = self.__create_tool_calls_message(tool_calls)
            messages.append(tool_calls_message)
            tool_messages = self.__execute_tool_calls(tool_calls_message.tool_calls)
            return await self.generate_answer(messages + tool_messages)

    async def __call_model(self, session: ClientSession, prompt: str):
        params = LlamaCppParams(prompt=prompt)

        async with session.post(self.model.vm_url, json=params.model_dump()) as response:
            # TODO: handle errors and retries
            if response.status == 200:
                response_data = await response.json()
                return response_data["content"]

    def __execute_tool_calls(self, tool_calls: list[MessageToolCall]) -> list[Message]:
        # TODO: support async function calls
        messages: list[Message] = []
        for call in tool_calls:
            function_name = call.function.name
            function_to_call = find(lambda x: x.__name__ == function_name, self.tools)
            if function_to_call is None:
                # TODO: handle error
                continue
            function_response = function_to_call(*call.function.arguments.values())
            messages.append(
                ToolResponseMessage(role=MessageRoleEnum.tool, name=function_name, tool_call_id=call.id,
                                    content=str(function_response)))
        return messages

    def __create_tool_calls_message(self, tool_calls: list[ToolCallFunction]) -> ToolCallMessage:
        return ToolCallMessage(role=MessageRoleEnum.assistant,
                               tool_calls=[MessageToolCall(type="function",
                                                           id=self.model.generate_tool_call_id(),
                                                           function=ToolCallFunction(name=call.name,
                                                                                     arguments=call.arguments)) for
                                           call in
                                           tool_calls])
