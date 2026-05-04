import os
from google.adk.agents import Agent

# ── Model selection ────────────────────────────────────────────────────────
_openai_base = os.getenv("OPENAI_API_BASE")
_openai_key = os.getenv("OPENAI_API_KEY")
_openai_model = os.getenv("OPENAI_MODEL", "gpt-4o")

if _openai_base and _openai_key:
    try:
        from google.adk.models.lite_llm import LiteLlm
    except ImportError:
        raise ImportError("OpenAI-compatible mode requires: pip install google-adk[extensions]")
    _model = LiteLlm(
        model=f"openai/{_openai_model}",
        api_base=_openai_base,
        api_key=_openai_key,
    )
    print(f"[agent] Using OpenAI-compatible endpoint: {_openai_base}, model: {_openai_model}")
else:
    _model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    print(f"[agent] Using Gemini API, model: {_model}")


# ── Langfuse client (shared) ───────────────────────────────────────────────

_lf = None

def _get_langfuse():
    global _lf
    if _lf is not None:
        return _lf
    host = os.getenv("LANGFUSE_HOST")
    secret = os.getenv("LANGFUSE_SECRET_KEY")
    public = os.getenv("LANGFUSE_PUBLIC_KEY")
    if not (host and secret and public):
        return None
    from langfuse import Langfuse
    _lf = Langfuse(secret_key=secret, public_key=public, host=host)
    return _lf


# ── Prompt from Langfuse ───────────────────────────────────────────────────

_DEFAULT_INSTRUCTION = (
    "You are a helpful insurance assistant. "
    "For any insurance-related questions (coverage, claims, policies, premiums, terms), "
    "you MUST use the retrieve_tool to search the knowledge base before answering. "
    "Always base your answer on the retrieved context when available. "
    "If the user asks to echo something, you MUST use the echo_tool. "
    "For non-insurance questions, answer directly without retrieval."
)

def _fetch_instruction() -> str:
    lf = _get_langfuse()
    if not lf:
        print("[agent] Langfuse env vars not set, using default instruction")
        return _DEFAULT_INSTRUCTION
    try:
        environment = os.getenv("LANGFUSE_ENVIRONMENT", "production")
        prompt = lf.get_prompt("kafka-agent-system", label=environment, cache_ttl_seconds=5)
        print(f"[agent] Fetched prompt from Langfuse (env={environment})")
        return prompt.compile()
    except Exception as e:
        print(f"[agent] Warning: could not fetch prompt from Langfuse ({e}), using default")
        return _DEFAULT_INSTRUCTION


_instruction = _fetch_instruction()


# ── Langfuse tracing callbacks ─────────────────────────────────────────────
# 一條 Kafka 訊息 = 一條 Langfuse trace
# 每次 LLM call（包含 tool call 後的 follow-up）= trace 下的一個 generation

from datetime import datetime, timezone

_pending_traces: dict = {}  # invocation_id → {trace, gen_count, gen_start_time, gen_input}


def _extract_text(contents) -> str:
    """從 llm_request.contents 取最後一筆可讀的文字。"""
    if not contents:
        return ""
    for content in reversed(contents):
        if not hasattr(content, "parts"):
            continue
        for part in content.parts:
            text = getattr(part, "text", None)
            if text:
                return text
            # tool result
            fn_resp = getattr(part, "function_response", None)
            if fn_resp:
                return f"[tool result] {getattr(fn_resp, 'response', '')}"
    return ""


def _before_model_callback(callback_context, llm_request):
    lf = _get_langfuse()
    if not lf:
        return None
    try:
        invocation_id = getattr(callback_context, "invocation_id", id(callback_context))
        gen_input = _extract_text(llm_request.contents)

        if invocation_id not in _pending_traces:
            # 第一次 LLM call：建立 trace，用原始 user input
            trace = lf.trace(
                name="kafka_agent",
                input=gen_input,
                metadata={"environment": os.getenv("LANGFUSE_ENVIRONMENT", "production")},
            )
            _pending_traces[invocation_id] = {"trace": trace, "gen_count": 0}

        state = _pending_traces[invocation_id]
        state["gen_start_time"] = datetime.now(timezone.utc)
        state["gen_input"] = gen_input
    except Exception as e:
        print(f"[agent] Langfuse before_model warning: {e}")
    return None


def _after_model_callback(callback_context, llm_response):
    lf = _get_langfuse()
    if not lf:
        return None
    try:
        invocation_id = getattr(callback_context, "invocation_id", id(callback_context))
        state = _pending_traces.get(invocation_id)
        if not state:
            return None

        state["gen_count"] += 1
        output_text = ""
        if llm_response.content and llm_response.content.parts:
            output_text = getattr(llm_response.content.parts[0], "text", "") or ""

        usage = getattr(llm_response, "usage_metadata", None)
        input_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
        output_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0
        model_name = _openai_model if (_openai_base and _openai_key) else "gemini-2.0-flash"

        state["trace"].generation(
            name=f"generate_content_{state['gen_count']}",
            model=model_name,
            input=state.get("gen_input", ""),
            output=output_text,
            start_time=state.get("gen_start_time"),
            end_time=datetime.now(timezone.utc),
            usage={"input": input_tokens, "output": output_tokens},
        )
        # 最終輸出更新到 trace 上
        state["trace"].update(output=output_text)
        lf.flush()
    except Exception as e:
        print(f"[agent] Langfuse after_model warning: {e}")
    return None


def _after_agent_callback(callback_context):
    """Agent 結束時清理 pending trace。"""
    invocation_id = getattr(callback_context, "invocation_id", id(callback_context))
    _pending_traces.pop(invocation_id, None)
    return None


# ── Tools ──────────────────────────────────────────────────────────────────

from agent.tools.retrieve import retrieve_tool


def echo_tool(message: str) -> dict:
    """Echo the input message back. Used to demonstrate tool call tracing.

    Args:
        message: The message to echo.

    Returns:
        A dict with the echoed message.
    """
    return {"echoed": message}


# ── Agent ──────────────────────────────────────────────────────────────────

root_agent = Agent(
    name="kafka_agent",
    model=_model,
    description="An agent that processes messages received from Kafka and responds to them.",
    instruction=_instruction,
    tools=[echo_tool, retrieve_tool],
    before_model_callback=_before_model_callback,
    after_model_callback=_after_model_callback,
    after_agent_callback=_after_agent_callback,
)
