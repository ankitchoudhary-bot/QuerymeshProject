# """Phase 1 tests: LLM abstraction contract and factory resolution."""

# from __future__ import annotations

# import pytest

# from querymesh.config import get_settings
# from querymesh.llm import LLMChunk, LLMConfigurationError, LLMResponse, get_llm
# from querymesh.llm.fake import FakeLLM


# @pytest.mark.unit
# def test_fake_llm_invoke_returns_llm_response() -> None:
#     llm = FakeLLM()
#     result = llm.invoke("hello world")

#     assert isinstance(result, LLMResponse)
#     assert result.text.startswith("[fake-response]")
#     assert "hello world" in result.text
#     assert result.model == "fake-echo-v1"
#     assert result.total_tokens == result.prompt_tokens + result.completion_tokens


# @pytest.mark.unit
# async def test_fake_llm_ainvoke_matches_invoke() -> None:
#     llm = FakeLLM()
#     sync_result = llm.invoke("async parity check")
#     async_result = await llm.ainvoke("async parity check")

#     assert sync_result.text == async_result.text
#     assert sync_result.model == async_result.model


# @pytest.mark.unit
# def test_fake_llm_stream_yields_chunks_and_ends_final() -> None:
#     llm = FakeLLM(response_text="one two three")
#     chunks = list(llm.stream("irrelevant prompt"))

#     assert len(chunks) == 3
#     assert all(isinstance(c, LLMChunk) for c in chunks)
#     assert chunks[-1].is_final is True
#     assert all(not c.is_final for c in chunks[:-1])
#     assert "".join(c.text for c in chunks) == "one two three"


# @pytest.mark.unit
# async def test_fake_llm_astream_matches_stream() -> None:
#     llm = FakeLLM(response_text="alpha beta")
#     sync_chunks = list(llm.stream("prompt"))
#     async_chunks = [c async for c in llm.astream("prompt")]

#     assert [c.text for c in sync_chunks] == [c.text for c in async_chunks]


# @pytest.mark.unit
# def test_factory_defaults_to_fake_provider() -> None:
#     get_settings.cache_clear()
#     llm = get_llm()

#     assert isinstance(llm, FakeLLM)
#     result = llm.invoke("factory default test")
#     assert isinstance(result, LLMResponse)


# @pytest.mark.unit
# def test_factory_rejects_unknown_provider() -> None:
#     from querymesh.config import Settings

#     settings = Settings(LLM_PROVIDER="not-a-real-provider")
#     with pytest.raises(LLMConfigurationError):
#         get_llm(settings)


# @pytest.mark.unit
# def test_factory_ollama_requires_model() -> None:
#     from querymesh.config import Settings

#     settings = Settings(LLM_PROVIDER="ollama", LLM_MODEL=None)
#     with pytest.raises(LLMConfigurationError):
#         get_llm(settings)


# @pytest.mark.unit
# def test_factory_ollama_constructs_with_model() -> None:
#     from querymesh.config import Settings
#     from querymesh.llm.ollama import OllamaLLM

#     settings = Settings(LLM_PROVIDER="ollama", LLM_MODEL="llama3")
#     llm = get_llm(settings)

#     assert isinstance(llm, OllamaLLM)
#     assert llm.model == "llama3"



"""Phase 1 tests: LLM abstraction contract and factory resolution."""

from __future__ import annotations

import pytest

from querymesh.config import get_settings
from querymesh.llm import LLMChunk, LLMConfigurationError, LLMResponse, get_llm
from querymesh.llm.fake import FakeLLM


@pytest.mark.unit
def test_fake_llm_invoke_returns_llm_response() -> None:
    llm = FakeLLM()
    result = llm.invoke("hello world")

    assert isinstance(result, LLMResponse)
    assert result.text.startswith("[fake-response]")
    assert "hello world" in result.text
    assert result.model == "fake-echo-v1"
    assert result.total_tokens == result.prompt_tokens + result.completion_tokens


@pytest.mark.unit
async def test_fake_llm_ainvoke_matches_invoke() -> None:
    llm = FakeLLM()
    sync_result = llm.invoke("async parity check")
    async_result = await llm.ainvoke("async parity check")

    assert sync_result.text == async_result.text
    assert sync_result.model == async_result.model


@pytest.mark.unit
def test_fake_llm_stream_yields_chunks_and_ends_final() -> None:
    llm = FakeLLM(response_text="one two three")
    chunks = list(llm.stream("irrelevant prompt"))

    assert len(chunks) == 3
    assert all(isinstance(c, LLMChunk) for c in chunks)
    assert chunks[-1].is_final is True
    assert all(not c.is_final for c in chunks[:-1])
    assert "".join(c.text for c in chunks) == "one two three"


@pytest.mark.unit
async def test_fake_llm_astream_matches_stream() -> None:
    llm = FakeLLM(response_text="alpha beta")
    sync_chunks = list(llm.stream("prompt"))
    async_chunks = [c async for c in llm.astream("prompt")]

    assert [c.text for c in sync_chunks] == [c.text for c in async_chunks]


@pytest.mark.unit
def test_factory_defaults_to_fake_provider(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    get_settings.cache_clear()
    llm = get_llm()

    assert isinstance(llm, FakeLLM)
    result = llm.invoke("factory default test")
    assert isinstance(result, LLMResponse)


@pytest.mark.unit
def test_factory_rejects_unknown_provider() -> None:
    from querymesh.config import Settings

    settings = Settings(LLM_PROVIDER="not-a-real-provider")
    with pytest.raises(LLMConfigurationError):
        get_llm(settings)


@pytest.mark.unit
def test_factory_ollama_requires_model() -> None:
    from querymesh.config import Settings

    settings = Settings(LLM_PROVIDER="ollama", LLM_MODEL=None)
    with pytest.raises(LLMConfigurationError):
        get_llm(settings)


@pytest.mark.unit
def test_factory_ollama_constructs_with_model() -> None:
    from querymesh.config import Settings
    from querymesh.llm.ollama import OllamaLLM

    settings = Settings(LLM_PROVIDER="ollama", LLM_MODEL="llama3")
    llm = get_llm(settings)

    assert isinstance(llm, OllamaLLM)
    assert llm.model == "llama3"
