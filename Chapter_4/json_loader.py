"""
json_loader.py
----------------
Loads the conversion/OCR team's JSON output (one JSON file per source
document, one entry per page) as ONE Document per PAGE.

Expected JSON shape (matches the conversion team's contract):
{
  "source_pdf": "02_FD_Product_Guide.pdf",
  "document_code": "FD-PROD-02",
  "version": "1.0",
  "pages": [
    {"page_number": 1, "text": "..."},
    {"page_number": 2, "text": "..."}
  ]
}

Why this loader exists at all: it replaces an in-house PDFLoader. Parsing
PDFs (tables, scanned pages, OCR) is a hard, specialized problem already
solved -- once, centrally -- by the conversion team. Consuming their JSON
means this pipeline has exactly ONE source of truth for "what does this
PDF actually say", instead of two PDF-parsing code paths that could
silently disagree.

NOTE: this loader deliberately does NOT judge OCR/extraction quality
(confidence thresholds, scanned-page detection, etc.) -- that is owned by
the conversion/OCR team, not this pipeline.
"""

import os
import glob
import json
from document import make_document

JSON_FOLDER = "../data/related_documents_json/"


def lazy_load_json(file_path: str):
    """Generator version -- yields one Document per page entry in the JSON."""
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    for page in data.get("pages", []):
        yield make_document(
            page_content=page.get("text", ""),
            metadata={
                "source": data.get("source_pdf", file_path),
                "document_code": data.get("document_code"),
                "version": data.get("version"),
                "page": page.get("page_number"),
            },
        )


def load_json(file_path: str) -> list:
    return list(lazy_load_json(file_path))


def load_all_json(folder: str = JSON_FOLDER) -> list:
    """Loads every .json file in a folder, concatenating all their Documents."""
    documents = []
    for path in sorted(glob.glob(os.path.join(folder, "*.json"))):
        documents.extend(load_json(path))
    return documents


if __name__ == "__main__":
    all_docs = load_all_json()
    print(f"Loaded {len(all_docs)} total pages across all JSON files")
    sources = sorted(set(d["metadata"]["source"] for d in all_docs))
    print(f"  Sources: {sources}")