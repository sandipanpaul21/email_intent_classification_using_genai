"""
incremental_ingestion.py
--------------------------
Skip re-embedding files that haven't changed since the last ingestion run --
the same discipline insert_fd_record()/update_fd_record() already established
for the FD database (fd_database.py), applied to source files instead of rows.

A "manifest" (a JSON file) remembers, for every tracked source file:
    { file_path: {"hash": <sha256 of file bytes>, "last_ingested": <UTC ISO timestamp>} }

Each run compares CURRENT file hashes against the manifest. Only NEW or
CHANGED files get (re)ingested -- unchanged files are skipped entirely:
no re-chunking, no re-embedding, no wasted API/embedding cost.

Known limitations (see theory section for the full discussion):
  - Single JSON file -> not safe for concurrent writers (last write wins).
  - Keyed on file PATH, not content -> a rename looks like "new file".
  - Manifest is updated in-memory and only persisted via save_manifest() --
    a crash mid-run loses ALL progress for that run (safe, but wasteful).
  - Does NOT track which embedding model produced existing vectors --
    swapping models silently leaves stale vectors with no warning.
"""

import os
import json
import hashlib
import glob
from datetime import datetime, timezone

MANIFEST_PATH = "ingestion_manifest.json"


def _file_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_manifest(path: str = MANIFEST_PATH) -> dict:
    """Reads our 'memory' of what's already been ingested. Empty dict on first run."""
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_manifest(manifest: dict, path: str = MANIFEST_PATH) -> None:
    """Writes the manifest back to disk so the NEXT run can read it."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def get_files_to_ingest(file_paths: list, manifest: dict) -> list:
    """Compares each file's CURRENT hash against the hash recorded last time.
    Different hash (or never seen before) = needs (re)ingesting."""
    to_ingest = []
    for path in file_paths:
        current_hash = _file_hash(path)
        if manifest.get(path, {}).get("hash") != current_hash:
            to_ingest.append(path)
    return to_ingest


def update_manifest_entry(manifest: dict, file_path: str) -> None:
    """Records 'this file, with this exact hash, was ingested just now'.
    Mutates manifest IN MEMORY ONLY -- call save_manifest() to persist."""
    manifest[file_path] = {
        "hash": _file_hash(file_path),
        "last_ingested": datetime.now(timezone.utc).isoformat(),
    }


if __name__ == "__main__":
    JSON_FOLDER = "../data/related_documents_json/"
    tracked_files = sorted(glob.glob(os.path.join(JSON_FOLDER, "*.json")))

    # ---- Run 1: everything is new ----
    manifest = load_manifest()
    to_ingest = get_files_to_ingest(tracked_files, manifest)
    print(f"Run 1 -- files to ingest: {[os.path.basename(f) for f in to_ingest]}")
    for f in to_ingest:
        update_manifest_entry(manifest, f)
    save_manifest(manifest)

    # ---- Run 2: nothing changed, expect empty list ----
    manifest = load_manifest()
    to_ingest = get_files_to_ingest(tracked_files, manifest)
    print(f"Run 2 -- files to ingest: {to_ingest}  (expect empty)")