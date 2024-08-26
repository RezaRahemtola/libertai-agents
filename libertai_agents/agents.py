import asyncio
from http import HTTPStatus
from typing import Callable, Awaitable, Any

import aiohttp
from aiohttp import ClientSession
from fastapi import APIRouter, FastAPI

from libertai_agents.interfaces.llamacpp import CustomizableLlamaCppParams, LlamaCppParams
from libertai_agents.interfaces.messages import Message, MessageRoleEnum, MessageToolCall, ToolCallFunction, \
    ToolCallMessage, ToolResponseMessage
from libertai_agents.interfaces.models import ModelInformation
from libertai_agents.models import Model
from libertai_agents.utils import find


class ChatAgent:
    model: Model
    system_prompt: str
    tools: list[Callable[..., Awaitable[Any]]]
    llamacpp_params: CustomizableLlamaCppParams
    app: FastAPI | None

    def __init__(self, model: Model, system_prompt: str, tools: list[Callable[..., Awaitable[Any]]] | None = None,
                 llamacpp_params: CustomizableLlamaCppParams = CustomizableLlamaCppParams(),
                 expose_api: bool = True):
        """
        Create a LibertAI chatbot agent that can answer to messages from users

        :param model: The LLM you want to use, selected from the available ones
        :param system_prompt: Customize the behavior of the agent with your own prompt
        :param tools: List of functions that the agent can call. Each function must be asynchronous, have a docstring and return a stringifyable response
        :param llamacpp_params: Override params given to llamacpp when calling the model
        :param expose_api: Set at False to avoid exposing an API (useful if you are using a custom trigger)
        """
        if tools is None:
            tools = []

        if len(set(map(lambda x: x.__name__, tools))) != len(tools):
            raise ValueError("Tool functions must have different names")
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools
        self.llamacpp_params = llamacpp_params

        if expose_api:
            # Define API routes
            router = APIRouter()
            router.add_api_route("/generate-answer", self.generate_answer, methods=["POST"])
            router.add_api_route("/model", self.get_model_information, methods=["GET"])

            self.app = FastAPI(title="LibertAI ChatAgent")
            self.app.include_router(router)

    def get_model_information(self) -> ModelInformation:
        """
        Get information about the model powering this agent
        """
        return ModelInformation(id=self.model.model_id, context_length=self.model.context_length)

    async def generate_answer(self, messages: list[Message]) -> str:
        """
        Generate an answer based on a conversation

        :param messages: List of messages previously sent in this conversation
        :return: The string response of the agent
        """
        if len(messages) == 0:
            raise ValueError("No previous message to respond to")
        if messages[-1].role not in [MessageRoleEnum.user, MessageRoleEnum.tool]:
            raise ValueError("Last message is not from the user or a tool response")

        prompt = self.model.generate_prompt(messages, self.system_prompt, self.tools)
        async with aiohttp.ClientSession() as session:
            response = await self.__call_model(session, prompt)

            tool_calls = self.model.extract_tool_calls_from_response(response)
            if len(tool_calls) == 0:
                return response

            tool_calls_message = self.__create_tool_calls_message(tool_calls)
            messages.append(tool_calls_message)
            executed_calls = self.__execute_tool_calls(tool_calls_message.tool_calls)
            results = await asyncio.gather(*executed_calls)
            tool_results_messages: list[Message] = [
                ToolResponseMessage(role=MessageRoleEnum.tool, name=call.function.name, tool_call_id=call.id,
                                    content=str(results[i])) for i, call in enumerate(tool_calls_message.tool_calls)]

            return await self.generate_answer(messages + tool_results_messages)

    async def __call_model(self, session: ClientSession, prompt: str):
        # TODO: support streaming - detect tools calls to avoid sending them as response
        params = LlamaCppParams(prompt=prompt, **self.llamacpp_params.model_dump())

        async with session.post(self.model.vm_url, json=params.model_dump()) as response:
            # TODO: handle errors and retries
            if response.status == HTTPStatus.OK:
                response_data = await response.json()
                return response_data["content"]

    def __execute_tool_calls(self, tool_calls: list[MessageToolCall]) -> list[Awaitable[Any]]:
        executed_calls: list[Awaitable[Any]] = []
        for call in tool_calls:
            function_name = call.function.name
            function_to_call = find(lambda x: x.__name__ == function_name, self.tools)
            if function_to_call is None:
                # TODO: handle error
                continue
            function_response = function_to_call(*call.function.arguments.values())
            executed_calls.append(function_response)

        return executed_calls

    def __create_tool_calls_message(self, tool_calls: list[ToolCallFunction]) -> ToolCallMessage:
        return ToolCallMessage(role=MessageRoleEnum.assistant,
                               tool_calls=[MessageToolCall(type="function",
                                                           id=self.model.generate_tool_call_id(),
                                                           function=ToolCallFunction(name=call.name,
                                                                                     arguments=call.arguments)) for
                                           call in
                                           tool_calls])
