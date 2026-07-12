import httpx
import respx

from app.providers.groq import GroqChatProvider

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


@respx.mock
def test_groq_complete_returns_message():
    respx.post(GROQ_URL).mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "Hi there"}}]},
        )
    )
    provider = GroqChatProvider(api_key="test", model="llama-3.3-70b-versatile")

    msg = provider.complete([{"role": "user", "content": "hello"}])

    assert msg["content"] == "Hi there"


@respx.mock
def test_groq_passes_tools_through():
    route = respx.post(GROQ_URL).mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": []}}]},
        )
    )
    provider = GroqChatProvider(api_key="test", model="llama-3.3-70b-versatile")

    provider.complete([{"role": "user", "content": "spend?"}], tools=[{"type": "function"}])

    sent = route.calls.last.request
    assert b'"tools"' in sent.content
