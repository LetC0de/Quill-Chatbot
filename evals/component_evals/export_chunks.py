"""
Export all stored chunks from the Qdrant vector store to JSON.

Produces chunks_dump.json — every chunk's id, text, and metadata — so you can
scan them (e.g. to hand-pick ideal_context for the faithfulness dataset).

Works with Qdrant cloud (same as the main project).
"""

import json
import os
import sys

# Add backend to path so we can import src modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", "backend", ".env"))

from qdrant_client import QdrantClient
from src.utils.settings import settings # type: ignore

COLLECTION_NAME = "documents"

# Connect to Qdrant
client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
    timeout=120.0,
)

# Scroll through all points in the collection
dump = []
offset = None
limit = 100

print(f"Exporting chunks from Qdrant collection '{COLLECTION_NAME}'...")

while True:
    points, next_offset = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=limit,
        offset=offset,
        with_payload=True,
        with_vectors=False,
    )

    if not points:
        break

    for point in points:
        payload = point.payload or {}
        dump.append({
            "id": point.id,
            "text": payload.get("page_content", ""),
            "meta": payload.get("metadata", {})
        })

    offset = next_offset
    if offset is None:
        break

    print(f"  Fetched {len(dump)} chunks so far...")

# Sort by document_id then by id so related chunks sit together
dump.sort(key=lambda c: (str(c["meta"].get("document_id", "")), c["id"]))

# Output path
output_path = os.path.join(os.path.dirname(__file__), "chunks_dump.json")

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(dump, f, indent=2, ensure_ascii=False)

print(f"\nDumped {len(dump)} chunks to {output_path}")

# Quick per-document count
from collections import Counter
counts = Counter(str(c["meta"].get("document_id", "?")) for c in dump)
for doc_id in sorted(counts):
    print(f"  document_id {doc_id}: {counts[doc_id]} chunks")