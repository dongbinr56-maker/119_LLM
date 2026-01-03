
"""rag/retriever.py

FAISS 인덱스를 로드하고, 질문을 임베딩하여 Top-k 근거 조각을 찾습니다.

중요 포인트:
- 인덱스는 normalize_embeddings=True로 생성했기 때문에,
  검색 점수는 코사인 유사도와 동일한 의미를 가집니다(정확히는 inner product == cosine).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import SETTINGS


@dataclass
class RetrievedChunk:
    chunk_id: str
    score: float
    page_start: int
    page_end: int
    section_path: str
    source: str
    text: str


class FaissRetriever:
    def __init__(self, artifacts_dir: Path, embedding_model_name: str = SETTINGS.embedding_model_name):
        self.artifacts_dir = artifacts_dir
        self.model = SentenceTransformer(embedding_model_name)

        # FAISS 인덱스 로드
        self.index = faiss.read_index(str(artifacts_dir / "faiss.index"))

        # chunks.jsonl 로드(텍스트 포함)
        self.records: List[Dict] = []
        chunks_path = artifacts_dir / "chunks.jsonl"
        with chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                self.records.append(json.loads(line))

    def _embed_query(self, query: str) -> np.ndarray:
        # E5 계열 권장: query prefix를 붙이면 검색 성능이 좋아지는 경우가 많음
        q = "query: " + query.strip()
        emb = self.model.encode([q], normalize_embeddings=True)
        return np.asarray(emb, dtype=np.float32)

    def search(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        qv = self._embed_query(query)
        scores, idxs = self.index.search(qv, top_k)

        out: List[RetrievedChunk] = []
        for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
            if idx < 0:
                continue
            rec = self.records[idx]
            out.append(RetrievedChunk(
                chunk_id=rec["chunk_id"],
                score=float(score),
                page_start=int(rec["page_start"]),
                page_end=int(rec["page_end"]),
                section_path=rec.get("section_path", "(auto)"),
                source=rec.get("source", ""),
                text=rec["text"],
            ))
        return out
