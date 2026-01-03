
"""rag/formatting.py

Streamlit UI에서 근거를 예쁘게 보여주기 위한 포맷터입니다.
"""

from __future__ import annotations

from typing import List, Dict
from dataclasses import asdict

from .retriever import RetrievedChunk


def citations_markdown(chunks: List[RetrievedChunk]) -> str:
    lines = []
    for i, ch in enumerate(chunks, start=1):
        lines.append(f"- 근거 {i}: {ch.source} p.{ch.page_start}-{ch.page_end} / score={ch.score:.3f} / {ch.section_path}")
    return "\n".join(lines)


def chunks_as_dicts(chunks: List[RetrievedChunk]) -> List[Dict]:
    return [asdict(c) for c in chunks]
