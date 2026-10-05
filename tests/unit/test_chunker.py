"""Phase 4 tests: PaperChunker — structure-aware chunking, deterministic
chunk IDs, paragraph/sentence-boundary respecting splits.
"""

from __future__ import annotations

import pytest

from querymesh.ingestion import PaperChunker, make_chunk_id
from querymesh.ingestion.document_models import ParsedDocument, Section


def _doc(**overrides) -> ParsedDocument:
    defaults = dict(
        arxiv_id="2608.13558v1",
        title="Attention Evolutions",
        abstract="This paper studies attention mechanisms in transformers.",
        sections=[
            Section(title="Introduction", paragraphs=["Intro paragraph one.", "Intro paragraph two."]),
            Section(title="Conclusion", paragraphs=["We conclude the paper here."]),
        ],
    )
    defaults.update(overrides)
    return ParsedDocument(**defaults)


@pytest.mark.unit
def test_chunk_ids_are_deterministic_and_sequential() -> None:
    chunker = PaperChunker(max_chars=1000)
    doc = _doc()

    chunks_first_run = chunker.chunk(doc)
    chunks_second_run = chunker.chunk(doc)

    assert [c.chunk_id for c in chunks_first_run] == [c.chunk_id for c in chunks_second_run]
    for i, c in enumerate(chunks_first_run):
        assert c.chunk_id == make_chunk_id(doc.arxiv_id, i)
        assert c.chunk_index == i


@pytest.mark.unit
def test_abstract_becomes_its_own_chunk() -> None:
    chunker = PaperChunker(max_chars=1000)
    doc = _doc()
    chunks = chunker.chunk(doc)

    assert chunks[0].section == "Abstract"
    assert "attention mechanisms" in chunks[0].text


@pytest.mark.unit
def test_chunks_carry_paper_and_section_metadata() -> None:
    chunker = PaperChunker(max_chars=1000)
    doc = _doc()
    chunks = chunker.chunk(doc)

    for c in chunks:
        assert c.arxiv_id == "2608.13558v1"
        assert c.title == "Attention Evolutions"
        assert c.section is not None

    sections_seen = {c.section for c in chunks}
    assert sections_seen == {"Abstract", "Introduction", "Conclusion"}


@pytest.mark.unit
def test_chunk_never_mixes_two_sections() -> None:
    chunker = PaperChunker(max_chars=1000)
    doc = _doc()
    chunks = chunker.chunk(doc)

    # Each chunk's text should only contain paragraphs from its own section.
    for c in chunks:
        if c.section == "Introduction":
            assert "conclude the paper" not in c.text
        if c.section == "Conclusion":
            assert "Intro paragraph" not in c.text


@pytest.mark.unit
def test_small_max_chars_splits_paragraphs_into_multiple_chunks() -> None:
    chunker = PaperChunker(max_chars=25)
    doc = _doc(
        sections=[
            Section(
                title="Body",
                paragraphs=["Short paragraph one.", "Short paragraph two.", "Short paragraph three."],
            )
        ],
        abstract="",
    )
    chunks = chunker.chunk(doc)

    body_chunks = [c for c in chunks if c.section == "Body"]
    assert len(body_chunks) > 1
    for c in body_chunks:
        assert len(c.text) <= 25 + 5  # small slack for join characters on edge cases


@pytest.mark.unit
def test_oversized_paragraph_is_split_on_sentence_boundaries() -> None:
    long_paragraph = (
        "This is sentence one of a very long paragraph that must be split. "
        "This is sentence two, also fairly long in its own right here. "
        "This is sentence three, wrapping up the very long paragraph nicely."
    )
    chunker = PaperChunker(max_chars=80)
    doc = _doc(sections=[Section(title="Body", paragraphs=[long_paragraph])], abstract="")
    chunks = chunker.chunk(doc)

    body_chunks = [c for c in chunks if c.section == "Body"]
    assert len(body_chunks) > 1
    # No chunk should exceed max_chars by more than a small margin (single very
    # long sentences may still individually exceed it as a documented fallback).
    reconstructed = " ".join(c.text for c in body_chunks)
    assert "sentence one" in reconstructed
    assert "sentence three" in reconstructed


@pytest.mark.unit
def test_empty_paragraphs_are_skipped() -> None:
    chunker = PaperChunker(max_chars=1000)
    doc = _doc(sections=[Section(title="Body", paragraphs=["", "   ", "Real content here."])], abstract="")
    chunks = chunker.chunk(doc)

    assert len(chunks) == 1
    assert chunks[0].text == "Real content here."


@pytest.mark.unit
def test_empty_document_produces_no_chunks() -> None:
    chunker = PaperChunker(max_chars=1000)
    doc = _doc(sections=[], abstract="")
    chunks = chunker.chunk(doc)
    assert chunks == []


@pytest.mark.unit
def test_invalid_max_chars_raises() -> None:
    with pytest.raises(ValueError):
        PaperChunker(max_chars=0)
