"""
Layer 2 trajectory tests using InMemoryRunner.
Verifies agent tool call decisions without AgentEvaluator (avoids is_final_response bug).
"""
import pytest
from google.adk.runners import InMemoryRunner
from google.genai import types


@pytest.fixture
def insurance_runner(test_agent):
    return InMemoryRunner(agent=test_agent, app_name="kafka_agent")


async def run_and_collect_tools(runner, text: str, session_id: str) -> list[str]:
    """Run agent and return list of tool names that were called."""
    session = await runner.session_service.create_session(
        app_name="kafka_agent", user_id="test_user"
    )
    tools_called = []
    async for event in runner.run_async(
        user_id="test_user",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text=text)]),
    ):
        # Collect tool call events
        if (
            event.content
            and event.content.parts
            and event.content.role == "model"
        ):
            for part in event.content.parts:
                fn_call = getattr(part, "function_call", None)
                if fn_call and getattr(fn_call, "name", None):
                    tools_called.append(fn_call.name)
    return tools_called


@pytest.mark.asyncio
async def test_insurance_premium_query_calls_retrieve_tool(insurance_runner):
    """保費問題 → retrieve_tool 必須被呼叫。"""
    tools = await run_and_collect_tools(
        insurance_runner,
        "30 歲男性投保長青守護保險，20 年繳，每月保費多少？",
        "session-ins-01",
    )
    assert "retrieve_tool" in tools, f"Expected retrieve_tool to be called, got: {tools}"


@pytest.mark.asyncio
async def test_claims_document_query_calls_retrieve_tool(insurance_runner):
    """理賠文件問題 → retrieve_tool 必須被呼叫。"""
    tools = await run_and_collect_tools(
        insurance_runner,
        "申請醫療實支實付理賠需要準備哪些文件？",
        "session-ins-02",
    )
    assert "retrieve_tool" in tools, f"Expected retrieve_tool to be called, got: {tools}"


@pytest.mark.asyncio
async def test_accident_insurance_query_calls_retrieve_tool(insurance_runner):
    """意外險問題 → retrieve_tool 必須被呼叫。"""
    tools = await run_and_collect_tools(
        insurance_runner,
        "意外險搭乘大眾運輸工具有什麼特別規定？",
        "session-ins-03",
    )
    assert "retrieve_tool" in tools, f"Expected retrieve_tool to be called, got: {tools}"


@pytest.mark.asyncio
async def test_greeting_does_not_call_retrieve_tool(insurance_runner):
    """問候語 → retrieve_tool 不應被呼叫。"""
    tools = await run_and_collect_tools(
        insurance_runner,
        "你好！",
        "session-gen-01",
    )
    assert "retrieve_tool" not in tools, f"Expected retrieve_tool NOT to be called, got: {tools}"


@pytest.mark.asyncio
async def test_general_question_does_not_call_retrieve_tool(insurance_runner):
    """非保險問題 → retrieve_tool 不應被呼叫。"""
    tools = await run_and_collect_tools(
        insurance_runner,
        "今天天氣怎麼樣？",
        "session-gen-02",
    )
    assert "retrieve_tool" not in tools, f"Expected retrieve_tool NOT to be called, got: {tools}"
