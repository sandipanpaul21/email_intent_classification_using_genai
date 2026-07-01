"""
dedup.py
---------
Two-stage duplicate detection for Documents coming from different filenames
(e.g. FD_Policy.json / FD_Policy_Final.json / FD_Policy_V2.json).

Stage 1 - HASH (cheap, exact):
    SHA-256 of the text. Identical content -> identical hash, regardless
    of filename. Runs on EVERY document; cost is negligible.

Stage 2 - COSINE SIMILARITY (expensive, near/fuzzy):
    Only runs on documents that SURVIVED stage 1. Embeddings cost real
    money/compute, so we never pay for them on content we'd already reject.
    Catches reworded duplicates: same fact, different phrasing.

Cost-ordering is the whole design: cheapest, most certain check first;
expensive, fuzzy check only on what's left.
"""

import hashlib
import numpy as np

# Loaded lazily so importing this module doesn't force a model download
# just to use content_hash() alone.
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _model


def content_hash(text: str) -> str:
    """A fingerprint for a piece of text. Identical text -> identical hash."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def find_duplicates(documents: list, near_duplicate_threshold: float = 0.97) -> list:
    """Returns a list of (doc_index, duplicate_of_index, 'exact' | 'near (score)')."""
    seen_hashes = {}          # hash -> index of the FIRST document with that hash
    kept_for_embedding = []   # (index, text) for docs that survived the hash check
    duplicates = []

    # ---- Stage 1: hash check (exact duplicates) ----
    for i, doc in enumerate(documents):
        h = content_hash(doc["page_content"])
        if h in seen_hashes:
            duplicates.append((i, seen_hashes[h], "exact"))
        else:
            seen_hashes[h] = i
            kept_for_embedding.append((i, doc["page_content"]))

    # ---- Stage 2: cosine similarity (near duplicates) ----
    if len(kept_for_embedding) > 1:
        indices, texts = zip(*kept_for_embedding)
        model = _get_model()
        embeddings = model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)

        already_flagged = set()
        for a in range(len(indices)):
            if indices[a] in already_flagged:
                continue
            for b in range(a + 1, len(indices)):
                if indices[b] in already_flagged:
                    continue
                score = float(np.dot(embeddings[a], embeddings[b]))  # vectors normalized -> dot == cosine
                if score >= near_duplicate_threshold:
                    duplicates.append((indices[b], indices[a], f"near ({score:.3f})"))
                    already_flagged.add(indices[b])

    return duplicates


if __name__ == "__main__":
    from document import make_document

    test_docs = [
        make_document("FD rate is 7.1 percent for 24 month tenure.", {"source": "FD_Policy.json"}),
        make_document("FD rate is 7.1 percent for 24 month tenure.", {"source": "FD_Policy_Final.json"}),
        make_document("FD rate is 7.1 percent for a 24 month tenure period.", {"source": "FD_Policy_V2.json"}),
        make_document("Premature withdrawal incurs a 1 percent penalty.", {"source": "FD_Policy.json"}),
    ]
    dupes = find_duplicates(test_docs, near_duplicate_threshold=0.85)
    for idx, of_idx, kind in dupes:
        print(f"  Document {idx} ({test_docs[idx]['metadata']['source']}) is a {kind} "
              f"duplicate of Document {of_idx} ({test_docs[of_idx]['metadata']['source']})")