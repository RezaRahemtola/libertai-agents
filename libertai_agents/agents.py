import aiohttp
from aiohttp import ClientSession

from libertai_agents.interfaces import Message, MessageRoleEnum, LlamaCppParams
from libertai_agents.models import Model


class ChatAgent:
    model: Model
    system_prompt: str
    tools: list

    def __init__(self, model: Model, system_prompt: str, tools: list | None = None):
        if tools is None:
            tools = []
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools

    async def generate_answer(self, messages: list[Message]) -> str:
        if len(messages) == 0:
            raise ValueError("No previous message to respond to")
        if messages[-1].role != MessageRoleEnum.user:
            raise ValueError("Last message is not from the user")

        prompt = self.model.generate_prompt(messages, self.system_prompt, self.tools)
        print(prompt)
        async with aiohttp.ClientSession() as session:
            return await self.__call_model(session, prompt)

    async def __call_model(self, session: ClientSession, prompt: str):
        params = LlamaCppParams(prompt=prompt)

        async with session.post(self.model.vm_url, json=params.model_dump()) as response:
            if response.status == 200:
                response_data = await response.json()
                return response_data["content"]
