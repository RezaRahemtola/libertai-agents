import asyncio

from libertai_agents.agents import ChatAgent
from libertai_agents.interfaces.messages import Message, MessageRoleEnum
from libertai_agents.models import get_model


async def get_current_temperature(location: str, unit: str) -> float:
    """
    Get the current temperature at a location.

    Args:
        location: The location to get the temperature for, in the format "City, Country"
        unit: The unit to return the temperature in. (choices: ["celsius", "fahrenheit"])
    Returns:
        The current temperature at the specified location in the specified units, as a float.
    """
    return 22.  # A real function should probably actually get the temperature!


agent = ChatAgent(model=get_model("NousResearch/Hermes-2-Pro-Llama-3-8B"),
                  system_prompt="You are a helpful assistant",
                  tools=[get_current_temperature], expose_api=False)


async def main():
    async for message in agent.generate_answer(
            [Message(role=MessageRoleEnum.user, content="What is the temperature in Paris and in Lyon?")]):
        print(message)


asyncio.run(main())
