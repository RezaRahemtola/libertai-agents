import asyncio

from libertai_agents.agents import ChatAgent
from libertai_agents.interfaces import Message, MessageRoleEnum
from libertai_agents.models import get_model
from libertai_agents.tools import get_current_temperature


async def start():
    agent = ChatAgent(model=get_model("mistralai/Mistral-Nemo-Instruct-2407"),
                      system_prompt="You are a helpful assistant",
                      tools=[get_current_temperature])
    response = await agent.generate_answer(
        [Message(role=MessageRoleEnum.user, content="What's the temperature in Paris and in Lyon in Celsius ?")])
    print(response)


asyncio.run(start())
