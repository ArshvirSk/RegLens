"""Chunking strategies.

Phase 0 defines the contract only. Implementations, in the order the PRD asks for them:

* ``fixed.py``      — Phase 1 baseline: fixed token windows with overlap.
* ``clause_aware.py`` — Phase 2: one chunk per numbered clause plus its heading path.
* ``speaker_aware.py`` — Phase 2: transcripts split on speaker turns / Q&A pairs.
* ``parent_child.py`` — Phase 2: small retrieval units backed by larger parents.
"""

from reglens.chunking.base import Chunk, Chunker, ParsedDocument, ParsedPage

__all__ = ["Chunk", "Chunker", "ParsedDocument", "ParsedPage"]
