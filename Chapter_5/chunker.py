"""
chunker.py
-----------
Sentence-aware chunking with configurable overlap.

Splits a Document's page_content into smaller Documents, never cutting
mid-sentence, and -- unlike a zero-overlap chunker -- carries forward the
last `overlap` sentences of each chunk into the start of the next one, so
an idea spanning a chunk boundary isn't fully lost to one side.
"""

import re
from document import make_document

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list:
    """Naive sentence splitter -- splits after ./!/? followed by whitespace.
    Good enough for structured policy/FAQ text; a real NLP sentence
    tokenizer (e.g. spaCy) would handle abbreviations more robustly."""
    sentences = SENTENCE_SPLIT_RE.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 1) -> list:
    """Sentence-aware chunking with overlap.

    chunk_size : max characters per chunk (soft limit -- a chunk stops
                 adding sentences once it would exceed this length).
    overlap    : number of trailing sentences from the previous chunk to
                 repeat at the start of the next chunk. 0 = no overlap
                 (the old, gap-having behavior).

    IMPORTANT: overlap is automatically capped so it can never consume an
    entire completed chunk. If it did, carrying back "the last N sentences"
    would just rebuild the identical chunk every time and the loop would
    never make progress -- a real infinite-loop bug, not just slowness.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks = []
    current = []
    current_len = 0
    i = 0

    while i < len(sentences):
        sentence = sentences[i]
        if current_len + len(sentence) > chunk_size and current:
            chunks.append(" ".join(current))

            # Cap overlap so at least ONE sentence is always dropped --
            # this is what guarantees forward progress.
            safe_overlap = min(overlap, len(current) - 1) if overlap > 0 else 0
            carry_back = current[-safe_overlap:] if safe_overlap > 0 else []

            current = list(carry_back)
            current_len = sum(len(s) for s in current)
            continue  # re-evaluate the same sentence against the new chunk
        current.append(sentence)
        current_len += len(sentence)
        i += 1

    if current:
        chunks.append(" ".join(current))

    return chunks


def chunk_document(doc: dict, chunk_size: int = 500, overlap: int = 1) -> list:
    """Splits one Document into several smaller Documents, preserving and
    extending metadata with a chunk index so each piece is still traceable
    back to its source/page/row."""
    pieces = chunk_text(doc["page_content"], chunk_size=chunk_size, overlap=overlap)
    result = []
    for idx, piece in enumerate(pieces):
        meta = dict(doc["metadata"])
        meta["chunk_index"] = idx
        result.append(make_document(page_content=piece, metadata=meta))
    return result


if __name__ == "__main__":
    sample = (
        "Premature withdrawal incurs a 1 percent penalty on the applicable rate. "
        "This does not apply if the FD is closed due to the death of the depositor. "
        "In such cases, the full contracted interest rate is paid up to the date of closure. "
        "Senior citizens receive an additional 0.5 percent interest on all tenures. "
        "This additional rate applies only to resident senior citizens aged 60 and above."
    )

    # chunk_size bumped to 180 so each chunk can actually fit 2 sentences --
    # at 120, no chunk could ever hold more than 1 sentence, so overlap had
    # nothing to carry forward and silently did nothing.
    CHUNK_SIZE = 180

    print("--- Zero overlap (the old gap) ---")
    for i, c in enumerate(chunk_text(sample, chunk_size=CHUNK_SIZE, overlap=0)):
        print(f"  Chunk {i}: {c}")

    print("\n--- With overlap=1 (the fix) ---")
    for i, c in enumerate(chunk_text(sample, chunk_size=CHUNK_SIZE, overlap=1)):
        print(f"  Chunk {i}: {c}")

    print("\n--- As Documents (chunk_document) ---")
    doc = make_document(sample, {"source": "fd_policy_demo.txt", "page": 1})
    for d in chunk_document(doc, chunk_size=CHUNK_SIZE, overlap=1):
        print(f"  {d['metadata']}: {d['page_content'][:60]!r}...")