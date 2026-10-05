"""Parser for arXiv's Atom XML search-response format."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime

from querymesh.ingestion.errors import ArxivError
from querymesh.ingestion.models import PaperMetadata

_ATOM_NS = "{http://www.w3.org/2005/Atom}"
_ARXIV_NS = "{http://arxiv.org/schemas/atom}"


def _text(el: ET.Element | None) -> str:
    if el is None or el.text is None:
        return ""
    return " ".join(el.text.split())


def _parse_datetime(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _extract_arxiv_id(id_url: str) -> str:
    # id_url looks like: http://arxiv.org/abs/2608.13558v1
    return id_url.rstrip("/").rsplit("/", 1)[-1]


def _parse_entry(entry: ET.Element) -> PaperMetadata:
    id_url = _text(entry.find(f"{_ATOM_NS}id"))
    title = _text(entry.find(f"{_ATOM_NS}title"))
    abstract = _text(entry.find(f"{_ATOM_NS}summary"))
    published = _parse_datetime(_text(entry.find(f"{_ATOM_NS}published")))
    updated = _parse_datetime(_text(entry.find(f"{_ATOM_NS}updated")))

    authors = [
        _text(author.find(f"{_ATOM_NS}name"))
        for author in entry.findall(f"{_ATOM_NS}author")
    ]
    authors = [a for a in authors if a]

    categories = [
        cat.get("term", "")
        for cat in entry.findall(f"{_ATOM_NS}category")
        if cat.get("term")
    ]

    pdf_url = None
    source_url = None
    for link in entry.findall(f"{_ATOM_NS}link"):
        rel = link.get("rel")
        title_attr = link.get("title")
        href = link.get("href")
        if title_attr == "pdf" and href:
            pdf_url = href
        elif rel == "alternate" and href:
            source_url = href

    doi_el = entry.find(f"{_ARXIV_NS}doi")
    doi = _text(doi_el) or None

    return PaperMetadata(
        arxiv_id=_extract_arxiv_id(id_url),
        title=title,
        authors=authors,
        abstract=abstract,
        categories=categories,
        published=published,
        updated=updated,
        pdf_url=pdf_url,
        source_url=source_url,
        doi=doi,
    )


def parse_feed(xml_text: str) -> list[PaperMetadata]:
    """Parse an arXiv Atom feed into a list of PaperMetadata."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ArxivError(f"Failed to parse arXiv response: {exc}") from exc

    entries = root.findall(f"{_ATOM_NS}entry")
    return [_parse_entry(entry) for entry in entries]
