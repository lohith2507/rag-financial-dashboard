import httpx

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqChatProvider:
    def __init__(self, api_key: str, model: str, timeout: float = 60.0):
        self._api_key = api_key
        self._model = model
        self._timeout = timeout

    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        payload: dict = {"model": self._model, "messages": messages, "temperature": 0.2}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        resp = httpx.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json=payload,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]
