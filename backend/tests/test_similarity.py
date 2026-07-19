import httpx
import pytest
import respx

from app.providers.nvidia import NVIDIA_URL, NvidiaEmbeddingProvider
from app.rag.similarity import cosine


def test_cosine_identical_vectors():
    assert cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors():
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_zero_vector_returns_zero():
    assert cosine([0.0, 0.0], [1.0, 2.0]) == 0.0


@respx.mock
def test_nvidia_provider_sends_query_input_type():
    route = respx.post(NVIDIA_URL).mock(
        return_value=httpx.Response(200, json={"data": [{"embedding": [0.1] * 1024, "index": 0}]})
    )
    provider = NvidiaEmbeddingProvider(api_key="k", model="m", input_type="query")
    provider.embed(["how much on food"])
    assert b'"input_type":"query"' in route.calls.last.request.content
