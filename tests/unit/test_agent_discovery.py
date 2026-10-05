# """Phase 15 tests: AgentRegistry (register/discover/get/list_agents) and
# build_default_registry wiring the five Phase 11 agents.
# """

# from __future__ import annotations

# import pytest

# from querymesh.a2a import (
#     AgentAlreadyRegisteredError,
#     AgentCard,
#     AgentNotFoundError,
#     AgentRegistry,
#     build_default_registry,
# )
# from querymesh.agents import CitationAgent, GraderAgent, ResearcherAgent, RetrieverAgent, SynthesizerAgent
# from querymesh.llm.fake import FakeLLM


# @pytest.mark.unit
# def test_register_and_get_by_name() -> None:
#     registry = AgentRegistry()
#     card = AgentCard(name="researcher", capabilities=["research"])
#     instance = object()

#     registry.register(card, instance)

#     assert registry.get("researcher") is instance


# @pytest.mark.unit
# def test_get_unknown_name_raises() -> None:
#     registry = AgentRegistry()
#     with pytest.raises(AgentNotFoundError):
#         registry.get("nonexistent")


# @pytest.mark.unit
# def test_register_duplicate_name_raises() -> None:
#     registry = AgentRegistry()
#     registry.register(AgentCard(name="researcher", capabilities=["research"]), object())

#     with pytest.raises(AgentAlreadyRegisteredError):
#         registry.register(AgentCard(name="researcher", capabilities=["research"]), object())


# @pytest.mark.unit
# def test_discover_by_capability() -> None:
#     registry = AgentRegistry()
#     researcher = object()
#     grader = object()

#     registry.register(AgentCard(name="researcher", capabilities=["research"]), researcher)
#     registry.register(AgentCard(name="grader", capabilities=["grading"]), grader)

#     assert registry.discover("research") == [researcher]
#     assert registry.discover("grading") == [grader]
#     assert registry.discover("nonexistent_capability") == []


# @pytest.mark.unit
# def test_discover_returns_multiple_agents_with_shared_capability() -> None:
#     registry = AgentRegistry()
#     a = object()
#     b = object()

#     registry.register(AgentCard(name="a", capabilities=["shared", "only_a"]), a)
#     registry.register(AgentCard(name="b", capabilities=["shared"]), b)

#     results = registry.discover("shared")
#     assert set(results) == {a, b}


# @pytest.mark.unit
# def test_list_agents_returns_all_cards() -> None:
#     registry = AgentRegistry()
#     registry.register(AgentCard(name="a", capabilities=["cap_a"]), object())
#     registry.register(AgentCard(name="b", capabilities=["cap_b"]), object())

#     cards = registry.list_agents()

#     assert {c.name for c in cards} == {"a", "b"}


# @pytest.mark.unit
# def test_empty_registry_list_and_discover_return_empty() -> None:
#     registry = AgentRegistry()
#     assert registry.list_agents() == []
#     assert registry.discover("anything") == []


# # --- build_default_registry -------------------------------------------


# @pytest.mark.unit
# def test_build_default_registry_registers_all_five_agents() -> None:
#     llm = FakeLLM()
#     researcher = ResearcherAgent(llm)
#     retriever = RetrieverAgent(retrieval_service=object())  # type: ignore[arg-type]
#     grader = GraderAgent(llm)
#     synthesizer = SynthesizerAgent(llm)
#     citation = CitationAgent()

#     registry = build_default_registry(researcher, retriever, grader, synthesizer, citation)

#     assert registry.get("researcher") is researcher
#     assert registry.get("retriever") is retriever
#     assert registry.get("grader") is grader
#     assert registry.get("synthesizer") is synthesizer
#     assert registry.get("citation") is citation

#     names = {c.name for c in registry.list_agents()}
#     assert names == {"researcher", "retriever", "grader", "synthesizer", "citation"}


# @pytest.mark.unit
# def test_build_default_registry_capabilities_match_spec() -> None:
#     llm = FakeLLM()
#     registry = build_default_registry(
#         ResearcherAgent(llm),
#         RetrieverAgent(retrieval_service=object()),  # type: ignore[arg-type]
#         GraderAgent(llm),
#         SynthesizerAgent(llm),
#         CitationAgent(),
#     )

#     assert registry.discover("research") == [registry.get("researcher")]
#     assert registry.discover("retrieval") == [registry.get("retriever")]
#     assert registry.discover("grading") == [registry.get("grader")]
#     assert registry.discover("synthesis") == [registry.get("synthesizer")]
#     assert registry.discover("citation") == [registry.get("citation")]



"""Phase 15 tests: AgentRegistry (register/discover/get/list_agents) and
build_default_registry wiring the five Phase 11 agents.
"""

from __future__ import annotations

import pytest

from querymesh.a2a import (
    AgentAlreadyRegisteredError,
    AgentCard,
    AgentNotFoundError,
    AgentRegistry,
    build_default_registry,
)
from querymesh.agents import CitationAgent, GraderAgent, ResearcherAgent, RetrieverAgent, SynthesizerAgent
from querymesh.llm.fake import FakeLLM


@pytest.mark.unit
def test_register_and_get_by_name() -> None:
    registry = AgentRegistry()
    card = AgentCard(name="researcher", capabilities=["research"])
    instance = object()

    registry.register(card, instance)

    assert registry.get("researcher") is instance


@pytest.mark.unit
def test_get_unknown_name_raises() -> None:
    registry = AgentRegistry()
    with pytest.raises(AgentNotFoundError):
        registry.get("nonexistent")


@pytest.mark.unit
def test_register_duplicate_name_raises() -> None:
    registry = AgentRegistry()
    registry.register(AgentCard(name="researcher", capabilities=["research"]), object())

    with pytest.raises(AgentAlreadyRegisteredError):
        registry.register(AgentCard(name="researcher", capabilities=["research"]), object())


@pytest.mark.unit
def test_discover_by_capability() -> None:
    registry = AgentRegistry()
    researcher = object()
    grader = object()

    registry.register(AgentCard(name="researcher", capabilities=["research"]), researcher)
    registry.register(AgentCard(name="grader", capabilities=["grading"]), grader)

    assert registry.discover("research") == [researcher]
    assert registry.discover("grading") == [grader]
    assert registry.discover("nonexistent_capability") == []


@pytest.mark.unit
def test_discover_returns_multiple_agents_with_shared_capability() -> None:
    registry = AgentRegistry()
    a = object()
    b = object()

    registry.register(AgentCard(name="a", capabilities=["shared", "only_a"]), a)
    registry.register(AgentCard(name="b", capabilities=["shared"]), b)

    results = registry.discover("shared")
    assert set(results) == {a, b}


@pytest.mark.unit
def test_list_agents_returns_all_cards() -> None:
    registry = AgentRegistry()
    registry.register(AgentCard(name="a", capabilities=["cap_a"]), object())
    registry.register(AgentCard(name="b", capabilities=["cap_b"]), object())

    cards = registry.list_agents()

    assert {c.name for c in cards} == {"a", "b"}


@pytest.mark.unit
def test_empty_registry_list_and_discover_return_empty() -> None:
    registry = AgentRegistry()
    assert registry.list_agents() == []
    assert registry.discover("anything") == []


# --- build_default_registry -------------------------------------------


@pytest.mark.unit
def test_build_default_registry_registers_all_five_agents() -> None:
    llm = FakeLLM()
    researcher = ResearcherAgent(llm)
    retriever = RetrieverAgent(retrieval_service=object())  # type: ignore[arg-type]
    grader = GraderAgent(llm)
    synthesizer = SynthesizerAgent(llm)
    citation = CitationAgent(llm)

    registry = build_default_registry(researcher, retriever, grader, synthesizer, citation)

    assert registry.get("researcher") is researcher
    assert registry.get("retriever") is retriever
    assert registry.get("grader") is grader
    assert registry.get("synthesizer") is synthesizer
    assert registry.get("citation") is citation

    names = {c.name for c in registry.list_agents()}
    assert names == {"researcher", "retriever", "grader", "synthesizer", "citation"}


@pytest.mark.unit
def test_build_default_registry_capabilities_match_spec() -> None:
    llm = FakeLLM()
    registry = build_default_registry(
        ResearcherAgent(llm),
        RetrieverAgent(retrieval_service=object()),  # type: ignore[arg-type]
        GraderAgent(llm),
        SynthesizerAgent(llm),
        CitationAgent(llm),
    )

    assert registry.discover("research") == [registry.get("researcher")]
    assert registry.discover("retrieval") == [registry.get("retriever")]
    assert registry.discover("grading") == [registry.get("grader")]
    assert registry.discover("synthesis") == [registry.get("synthesizer")]
    assert registry.discover("citation") == [registry.get("citation")]
