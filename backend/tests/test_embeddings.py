import httpx
import respx

from app.providers.nvidia import NvidiaEmbeddingProvider

NVIDIA_URL = "https://integrate.api.nvidia.com/v1/embeddings"


@respx.mock
def test_nvidia_embed_returns_vectors():
    respx.post(NVIDIA_URL).mock(
        return_value=httpx.Response(
            200,
            json={"data": [
                {"embedding": [0.1] * 1024, "index": 0},
                {"embedding": [0.2] * 1024, "index": 1},
            ]},
        )
    )
    provider = NvidiaEmbeddingProvider(api_key="test", model="nvidia/nv-embedqa-e5-v5")

    vectors = provider.embed(["hello", "world"])

    assert len(vectors) == 2
    assert len(vectors[0]) == 1024
    assert vectors[1][0] == 0.2
