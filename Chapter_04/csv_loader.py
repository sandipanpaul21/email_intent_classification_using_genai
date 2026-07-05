"""
csv_loader.py
--------------
Loads tabular files (fd_dataset_messy.csv, fd_master_database.csv) as ONE
Document per ROW. Every column is folded into page_content as "key: value"
lines, and the row index is preserved in metadata so any later debugging
can trace a Document straight back to its row in the source file.

Design choice: row-level granularity, not whole-file. A customer's FD
record is the natural retrieval unit -- you want a search to return "this
one customer's record", not "the entire 20,000-row table".
"""

import csv
from email_intent_classification_using_genai.Chapter_5.document import make_document


def lazy_load_csv(file_path: str):
    """Generator version -- reads row by row, never holds the whole file in memory."""
    with open(file_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)          # each row -> {column_name: value}
        for i, row in enumerate(reader):
            content = "\n".join(f"{k}: {v}" for k, v in row.items())
            yield make_document(
                page_content=content,
                metadata={"source": file_path, "row": i},
            )


def load_csv(file_path: str) -> list:
    """Eager version -- use only when the file is small enough to fit in memory."""
    return list(lazy_load_csv(file_path))


if __name__ == "__main__":
    docs = load_csv("../data/fd_master_database.csv")
    print(f"Loaded {len(docs)} rows as Documents")
    print(f"  Document 0 metadata     : {docs[0]['metadata']}")
    print(f"  Document 0 content[:120]: {docs[0]['page_content'][:120]!r}")