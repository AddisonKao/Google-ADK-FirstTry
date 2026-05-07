"""
Agent module for trajectory tests.
Uses a mock retrieve_tool function — no real DB connection needed.
The mock EXECUTES (not intercepted), so AgentEvaluator counts it in trajectory.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from google.adk.agents import Agent
from agent.agent import _model, _DEFAULT_INSTRUCTION
from tests.conftest import INSURANCE_FIXTURES, DEFAULT_FIXTURE


def retrieve_tool(query: str) -> str:
    """Search the insurance knowledge base and return relevant context.

    Args:
        query: The question or topic to search for in the knowledge base.

    Returns:
        A string containing the most relevant passages from the knowledge base.
    """
    q = query.lower()
    for keyword, response in INSURANCE_FIXTURES.items():
        if keyword in q:
            return response
    return DEFAULT_FIXTURE


root_agent = Agent(
    name="kafka_agent",
    model=_model,
    description="An agent that processes messages received from Kafka and responds to them.",
    instruction=_DEFAULT_INSTRUCTION,
    tools=[retrieve_tool],
)

# ADK AgentEvaluator requires either module.agent or module name ending with .agent
agent = root_agent
