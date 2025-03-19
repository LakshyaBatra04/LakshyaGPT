from agno.agent import Agent
from agno.tools.duckduckgo import DuckDuckGoTools

agent = Agent(tools=[DuckDuckGoTools()], show_tool_calls=True)
agent.print_response("Tell me about the russia ukraine war situation?", markdown=True)