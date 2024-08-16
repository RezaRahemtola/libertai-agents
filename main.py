import asyncio

from libertai_agents.agents import ChatAgent
from libertai_agents.interfaces import Message, MessageRoleEnum
from libertai_agents.models import Hermes2Pro
from libertai_agents.tools import get_current_temperature


async def start():
    agent = ChatAgent(model=Hermes2Pro, system_prompt="You are a helpful assistant", tools=[get_current_temperature])
    response = await agent.generate_answer(
        [Message(role=MessageRoleEnum.user, content="What's the temperature in Paris ?")])
    print(response)


asyncio.run(start())
