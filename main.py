from libertai_agents.agents import ChatAgent
from libertai_agents.models import get_model
from libertai_agents.tools import get_current_temperature

agent = ChatAgent(model=get_model("NousResearch/Hermes-2-Pro-Llama-3-8B"),
                  system_prompt="You are a helpful assistant",
                  tools=[get_current_temperature])

app = agent.app
