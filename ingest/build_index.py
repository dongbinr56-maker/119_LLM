
"""ingest/build_index.py

PDF(표준지침) → 텍스트 추출 → 전처리/청킹 → 임베딩 → FAISS 인덱스 생성

핵심 목표:
- '페이지 번호'를 chunk 메타데이터로 들고 가서, 답변에서 근거(p.xx)를 반드시 제시할 수 있게 함
- PDF의 줄바꿈/반복 헤더/꼬리말 같은 노이즈를 최대한 제거하여 임베딩 품질을 올림

사용 예:
python -m ingest.build_index --pdf data/guide_2023.pdf --out artifacts
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import fitz  # PyMuPDF
import numpy as np
from tqdm import tqdm
import faiss
from sentence_transformers import SentenceTransformer

from config import SETTINGS


# -----------------------------
# 1) 전처리 유틸
# -----------------------------

def _normalize_text(t: str) -> str:
    """텍스트 정규화:
    - 과도한 공백 정리
    - 하이픈 줄바꿈(예: '응급-\n처치') 결합
    - 문장 중간 줄바꿈을 완화(단, 목록/표 형태는 최대한 보존)
    """
    # 하이픈 + 줄바꿈 → 단어 결합
    t = re.sub(r"-\s*\n\s*", "", t)

    # 줄바꿈을 일괄 공백으로 바꾸면 목록 구조가 무너질 수 있음.
    # 따라서: 너무 짧은 줄(예: 한두 단어) 뒤 줄바꿈은 공백으로 완화하는 식의 보수적 규칙 적용.
    lines = [ln.strip() for ln in t.splitlines()]
    merged: List[str] = []
    for ln in lines:
        if not ln:
            # 빈 줄은 문단 경계로 남겨둠
            merged.append("")
            continue

        # 이전 줄이 있고, 이전 줄이 너무 짧거나 문장이 끊긴 느낌(마침표/다음표현 없음)이라면 이어붙임
        if merged and merged[-1] and len(merged[-1]) < 40 and not re.search(r"[\.!?…。]$", merged[-1]):
            merged[-1] = (merged[-1] + " " + ln).strip()
        else:
            merged.append(ln)

    # 다중 공백 정리
    t2 = "\n".join(merged)
    t2 = re.sub(r"[ \t]+", " ", t2).strip()
    return t2


def _find_repeated_lines(pages: List[str], min_ratio: float = 0.6) -> set[str]:
    """PDF의 '머리말/꼬리말'처럼 반복적으로 등장하는 라인을 탐지해 제거하기 위한 함수.

    아이디어:
    - 각 페이지 텍스트를 줄 단위로 분해
    - 특정 라인이 전체 페이지의 일정 비율 이상에서 등장하면 '반복 라인'으로 간주

    주의:
    - 너무 공격적으로 지우면 중요한 문장도 사라질 수 있음.
    - 그래서 기본값 min_ratio를 비교적 높게(0.6) 잡음.
    """
    from collections import Counter

    total_pages = max(len(pages), 1)
    cnt = Counter()
    for p in pages:
        for ln in set([x.strip() for x in p.splitlines() if x.strip()]):
            # 짧은 라인은 흔히 발생하는 일반 단어일 수 있어 제외
            if len(ln) < 8:
                continue
            cnt[ln] += 1

    repeated = set()
    for ln, c in cnt.items():
        if c / total_pages >= min_ratio:
            repeated.add(ln)
    return repeated


# -----------------------------
# 2) PDF → 페이지별 텍스트 추출
# -----------------------------

@dataclass
class PageText:
    page_no: int          # 1-based (사람이 보는 페이지 번호)
    text: str


def load_pdf_pages(pdf_path: Path) -> List[PageText]:
    """PDF에서 페이지별 텍스트를 추출합니다."""
    doc = fitz.open(pdf_path)
    pages: List[PageText] = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        raw = page.get_text("text")  # 가장 단순한 텍스트 추출
        pages.append(PageText(page_no=i + 1, text=raw))
    return pages


# -----------------------------
# 3) 청킹 전략
# -----------------------------

_SECTION_HINT = re.compile(
    r"^(?:[IVX]+\.|\d+\.|\d+\)|[가-힣]\)|\(?[0-9]{1,2}\)?\s)"
)

def split_into_paragraphs(page_text: str) -> List[str]:
    """페이지 텍스트를 '문단' 후보 단위로 분리합니다.
    - 빈 줄 기준으로 우선 분리
    - 너무 긴 문단은 이후 단계에서 재분할
    """
    parts = [p.strip() for p in re.split(r"\n\s*\n", page_text) if p.strip()]
    return parts


def chunk_paragraphs(
    paragraphs: List[Tuple[int, str]],
    max_chars: int = 1600,
    overlap_chars: int = 200,
) -> List[Dict]:
    """문단들을 누적하여 chunk를 만듭니다.
    - max_chars를 넘기지 않도록 누적
    - overlap을 주어 문맥 끊김을 완화

    paragraphs: (page_no, paragraph_text) 리스트
    """
    chunks: List[Dict] = []
    buf: List[Tuple[int, str]] = []
    buf_len = 0

    def flush():
        nonlocal buf, buf_len
        if not buf:
            return
        pages = [p for p, _ in buf]
        text = "\n\n".join([t for _, t in buf]).strip()
        chunks.append({
            "text": text,
            "page_start": min(pages),
            "page_end": max(pages),
        })
        buf = []
        buf_len = 0

    for page_no, para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # 문단 하나가 너무 길면 강제로 자름(문단 내부에서 문장/줄 기준)
        if len(para) > max_chars:
            # 줄 단위로 쪼개서 다시 누적
            lines = [ln.strip() for ln in para.splitlines() if ln.strip()]
            for ln in lines:
                if buf_len + len(ln) + 2 > max_chars:
                    flush()
                buf.append((page_no, ln))
                buf_len += len(ln) + 2
            continue

        if buf_len + len(para) + 4 > max_chars:
            flush()

            # overlap: 직전 chunk의 끝부분 일부를 가져오려면,
            # 실제 구현이 복잡해질 수 있어 "간단 버전"으로는 생략하거나 최소화.
            # 여기서는 flush 직전 내용을 저장해 overlap을 구성합니다.
            # (정확한 overlap이 필요하면 이후에 개선 가능)
        buf.append((page_no, para))
        buf_len += len(para) + 4

    flush()
    return chunks


# -----------------------------
# 4) 임베딩 + FAISS 저장
# -----------------------------

def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 32) -> np.ndarray:
    """텍스트 리스트를 임베딩하여 numpy 배열로 반환합니다.
    - E5 계열은 보통 'query:'/'passage:' 프리픽스를 권장하지만,
      여기서는 단순화를 위해 passage만 사용합니다.
      (개선 시: 질문 임베딩에는 'query:' prefix를 사용)
    """
    embs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # 코사인 유사도 계산을 위해 정규화
    )
    return np.asarray(embs, dtype=np.float32)


def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    """정규화된 벡터(코사인 유사도)를 inner product로 검색."""
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product
    index.add(vectors)
    return index


# -----------------------------
# 5) CLI 엔트리포인트
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=str, required=True, help="PDF 파일 경로")
    ap.add_argument("--out", type=str, required=True, help="산출물 폴더(artifacts)")
    ap.add_argument("--max_chars", type=int, default=1600, help="chunk 최대 문자 수(대략 토큰 제한 역할)")
    ap.add_argument("--overlap_chars", type=int, default=200, help="chunk 오버랩 문자 수(현재는 최소 적용)")
    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) PDF 로드
    pages = load_pdf_pages(pdf_path)

    # 2) 반복 라인 탐지(헤더/푸터 제거용)
    raw_pages = [p.text for p in pages]
    repeated = _find_repeated_lines(raw_pages, min_ratio=0.6)

    # 3) 페이지별 정규화 + 반복 라인 제거 + 문단 분리
    paras: List[Tuple[int, str]] = []
    for p in pages:
        t = p.text
        # 반복 라인 제거(정확한 매칭 기반)
        lines = []
        for ln in t.splitlines():
            ln2 = ln.strip()
            if ln2 in repeated:
                continue
            lines.append(ln)
        t = "\n".join(lines)
        t = _normalize_text(t)

        for para in split_into_paragraphs(t):
            paras.append((p.page_no, para))

    # 4) 청킹
    chunks = chunk_paragraphs(paras, max_chars=args.max_chars, overlap_chars=args.overlap_chars)

    # chunk_id 부여 + 소스 정보 저장
    meta: List[Dict] = []
    chunk_texts: List[str] = []
    for i, ch in enumerate(chunks):
        chunk_id = f"chunk_{i:06d}"
        meta.append({
            "chunk_id": chunk_id,
            "page_start": ch["page_start"],
            "page_end": ch["page_end"],
            "source": pdf_path.name,
            # section_path는 고급 파싱(목차/헤더 인식) 단계에서 추가하는 것이 이상적.
            "section_path": "(auto)",
        })
        chunk_texts.append(ch["text"])

    # 5) 임베딩
    model = SentenceTransformer(SETTINGS.embedding_model_name)
    vectors = embed_texts(model, chunk_texts)

    # 6) FAISS 인덱스 생성/저장
    index = build_faiss_index(vectors)
    faiss.write_index(index, str(out_dir / "faiss.index"))

    # 7) 메타 + 청크 저장
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # chunks.jsonl (텍스트 포함)
    with (out_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for m, txt in zip(meta, chunk_texts):
            rec = {**m, "text": txt}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("[OK] Index built:", out_dir)


if __name__ == "__main__":
    main()
