import httpx

NVIDIA_URL = "https://integrate.api.nvidia.com/v1/embeddings"


class NvidiaEmbeddingProvider:
    def __init__(self, api_key: str, model: str, timeout: float = 30.0):
        self._api_key = api_key
        self._model = model
        self._timeout = timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = httpx.post(
            NVIDIA_URL,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"input": texts, "model": self._model, "input_type": "passage"},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        data = sorted(resp.json()["data"], key=lambda d: d["index"])
        return [d["embedding"] for d in data]
