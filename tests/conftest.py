"""Shared pytest fixtures for ADK agent tests."""
import os
import pytest

# Ensure no real external connections during tests
os.environ.setdefault("LANGFUSE_HOST", "")
os.environ.setdefault("LANGFUSE_SECRET_KEY", "")
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "")
os.environ.setdefault("DATABASE_URL", "postgresql://langfuse:langfuse@localhost:5432/langfuse")

# Insurance knowledge fixture data (simulates pgvector results)
INSURANCE_FIXTURES = {
    "月繳": "30 歲男性，保額 100 萬，20 年繳，每月保費約 2,850 元。",
    "保費": "30 歲男性，保額 100 萬，20 年繳，每月保費約 2,850 元。",
    "理賠": "理賠申請所需文件：理賠申請書（TPS-F003）、診斷證明書正本、醫療費用收據正本。",
    "文件": "理賠申請所需文件：理賠申請書（TPS-F003）、診斷證明書正本、醫療費用收據正本。",
    "意外": "意外傷害險第一類職業（內勤文職）年繳保費為 2,100 元（保額 200 萬）。",
    "大眾運輸": "搭乘大眾運輸工具時，意外身故或失能保險金加倍給付（條款第 TPS-ACC-012 條）。",
    "除外": "除外不保事項包含：故意自殘、戰爭、核子輻射、犯罪行為導致之傷亡。",
}

DEFAULT_FIXTURE = "根據保險條款，相關資訊如下：太平盛世人壽提供多種保險產品。"


def mock_retrieve_callback(tool, args, tool_context):
    """
    ADK before_tool_callback: intercepts retrieve_tool and returns fixture data.
    Returns non-None to skip real tool execution (no pgvector connection needed).
    """
    if tool.name == "retrieve_tool":
        query = args.get("query", "").lower()
        for keyword, response in INSURANCE_FIXTURES.items():
            if keyword in query:
                return response
        return DEFAULT_FIXTURE
    return None  # allow other tools to run normally


@pytest.fixture
def test_agent():
    """Agent wired with mock retrieve_tool callback (no real DB needed)."""
    from google.adk.agents import Agent
    from agent.agent import _model, _DEFAULT_INSTRUCTION
    from agent.tools.retrieve import retrieve_tool

    return Agent(
        name="kafka_agent",
        model=_model,
        instruction=_DEFAULT_INSTRUCTION,
        tools=[retrieve_tool],
        before_tool_callback=mock_retrieve_callback,
    )


@pytest.fixture
def runner(test_agent):
    """InMemoryRunner with test agent — no Kafka, no DB."""
    from google.adk.runners import InMemoryRunner
    return InMemoryRunner(agent=test_agent, app_name="kafka_agent")
