"""
document.py
-----------
The single shared shape every loader in this pipeline returns.

A "Document" is just a dict with two keys:
  - page_content : str   -> the actual text
  - metadata     : dict  -> where it came from (source file, page no., row no., ...)

Why this exists: every downstream stage (dedup, chunking, embedding, search)
is written ONCE against this shape, regardless of whether the text originally
came from a .txt, .csv, .xlsx, or .json file. Change the source format,
and only the loader changes -- nothing downstream needs to know.
"""

from typing import Optional


def make_document(page_content: str, metadata: Optional[dict] = None) -> dict:
    """Build one Document. metadata defaults to an empty dict, never None,
    so downstream code can always safely do doc["metadata"].get(...)."""
    return {"page_content": page_content, "metadata": metadata or {}}


if __name__ == "__main__":
    doc = make_document("FD rate is 7.1% for 24 months.", {"source": "demo.txt"})
    print(doc)