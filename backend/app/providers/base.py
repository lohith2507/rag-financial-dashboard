from typing import Protocol


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class ChatProvider(Protocol):
    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        ...
