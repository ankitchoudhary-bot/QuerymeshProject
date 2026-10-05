"""Sample arXiv Atom XML fixtures used by ingestion tests."""

ENTRY_TEMPLATE = """  <entry>
    <id>http://arxiv.org/abs/{arxiv_id}</id>
    <updated>2026-06-10T12:00:00Z</updated>
    <published>2026-06-09T09:00:00Z</published>
    <title>{title}</title>
    <summary>{summary}</summary>
    <author><name>{author}</name></author>
    <arxiv:doi xmlns:arxiv="http://arxiv.org/schemas/atom">10.1000/{arxiv_id}</arxiv:doi>
    <link href="http://arxiv.org/abs/{arxiv_id}" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/{arxiv_id}" rel="related" type="application/pdf"/>
    <category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
"""

FEED_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>ArXiv Query</title>
{entries}</feed>
"""


def make_feed(entries: list[dict]) -> str:
    entries_xml = "".join(ENTRY_TEMPLATE.format(**entry) for entry in entries)
    return FEED_TEMPLATE.format(entries=entries_xml)


def make_entry(arxiv_id: str, title: str = "A Test Paper", author: str = "Jane Doe") -> dict:
    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "summary": f"This is the abstract for {title}.",
        "author": author,
    }


EMPTY_FEED = FEED_TEMPLATE.format(entries="")
