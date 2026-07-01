"""
recursive_text_splitter.py
-----------------------------
RecursiveCharacterTextSplitter mechanics: tries separators in priority
order -- paragraph -> line -> space -> character -- recursively falling
back to the next separator only when the current one can't produce a
small-enough piece.

Unlike CharacterTextSplitter (Topic 4), which commits to ONE separator and
hard-cuts the instant it fails, this tries progressively less-preserving
splits before ever resorting to a blind character cut.
"""

from document import make_document

DEFAULT_SEPARATORS = ["\n\n", "\n", " ", ""]  # paragraph, line, space, character


def _split_with_separator(text: str, separator: str) -> list:
    if separator == "":
        return list(text)  # character-level: every character is its own piece
    return text.split(separator)


def _recursive_split(text: str, separators: list, chunk_size: int) -> list:
    """Core recursive logic: try the first separator; for any piece still
    too large, recurse into it with the NEXT separator down the list."""
    if len(text) <= chunk_size:
        return [text]

    separator = separators[0]
    remaining_separators = separators[1:]
    pieces = _split_with_separator(text, separator)

    final_pieces = []
    for piece in pieces:
        if len(piece) <= chunk_size:
            final_pieces.append(piece)
        elif remaining_separators:
            # this piece is still too big -- recurse with the next separator
            final_pieces.extend(_recursive_split(piece, remaining_separators, chunk_size))
        else:
            # no separators left -- absolute last resort, hard character cut
            final_pieces.extend(
                piece[i:i + chunk_size] for i in range(0, len(piece), chunk_size)
            )

    return final_pieces


def _merge_small_pieces(pieces: list, separator: str, chunk_size: int) -> list:
    """Greedily packs adjacent small pieces back together up to chunk_size,
    so we don't emit one chunk per tiny line/sentence when several fit
    comfortably together."""
    merged, current = [], ""
    for piece in pieces:
        candidate = (current + separator + piece) if current else piece
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                merged.append(current)
            current = piece
    if current:
        merged.append(current)
    return merged


def recursive_character_split(text: str, chunk_size: int = 200, separators: list = None) -> list:
    separators = separators or DEFAULT_SEPARATORS
    raw_pieces = _recursive_split(text, separators, chunk_size)
    # merge using the broadest separator that's still safe (paragraph break)
    return _merge_small_pieces(raw_pieces, separators[0], chunk_size)


if __name__ == "__main__":
    sample = (
        "Premature withdrawal incurs a 1 percent penalty on the applicable rate.\n\n"
        "This does not apply if the FD is closed due to the death of the depositor. "
        "In such cases, the full contracted interest rate is paid up to the date of closure.\n\n"
        "Senior citizens receive an additional 0.5 percent interest on all tenures. "
        "This additional rate applies only to resident senior citizens aged 60 and above."
    )

    print("--- Default separator order (paragraph -> line -> space -> char) ---")
    for i, c in enumerate(recursive_character_split(sample, chunk_size=120)):
        print(f"  Chunk {i} ({len(c)} chars): {c!r}")

    print("\n--- Custom separator order (forces deeper fallback for comparison) ---")
    for i, c in enumerate(recursive_character_split(sample, chunk_size=40, separators=["\n\n", " ", ""])):
        print(f"  Chunk {i} ({len(c)} chars): {c!r}")