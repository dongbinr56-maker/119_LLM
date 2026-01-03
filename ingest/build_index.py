"""
ingest/build_index.py

이 파일은 PDF 문서를 읽어서 검색 가능한 형태(FAISS 인덱스)로 변환하는 스크립트입니다.

처리 과정:
PDF(표준지침) → 텍스트 추출 → 전처리/청킹 → 임베딩 → FAISS 인덱스 생성

핵심 목표:
- '페이지 번호'를 chunk 메타데이터로 유지하여, 답변에서 근거(p.xx)를 반드시 제시할 수 있게 함
- PDF의 줄바꿈/반복 헤더/꼬리말 같은 노이즈를 최대한 제거하여 임베딩 품질을 향상시킴

사용 예:
python -m ingest.build_index --pdf data/guide_2023.pdf --out artifacts

이 스크립트를 실행하면 artifacts 폴더에 다음 파일들이 생성됩니다:
- faiss.index: FAISS 검색 인덱스 파일
- meta.json: 문서 조각들의 메타데이터 (페이지 번호, 파일명 등)
- chunks.jsonl: 문서 조각들의 텍스트와 메타데이터 (JSON Lines 형식)
"""

from __future__ import annotations  # 미래 버전 Python 호환성

import argparse  # 명령줄 인자 파싱용
import json  # JSON 파일 읽기/쓰기용
import re  # 정규표현식 (텍스트 패턴 매칭/변환용)
from dataclasses import dataclass  # 데이터클래스 정의용
from pathlib import Path  # 파일 경로 다루기용
from typing import Dict, Iterable, List, Tuple  # 타입 힌팅용

import fitz  # PyMuPDF: PDF 파일을 읽고 텍스트를 추출하는 라이브러리
import numpy as np  # 수치 계산용 라이브러리 (벡터 연산 등)
from tqdm import tqdm  # 진행 상황 표시용 (진행 바 표시)
import faiss  # Facebook의 빠른 검색 라이브러리 (벡터 검색 인덱스 생성/저장)
from sentence_transformers import SentenceTransformer  # 텍스트를 벡터(임베딩)로 변환하는 모델

from config import SETTINGS  # 프로젝트 설정 (임베딩 모델 이름 등)


# -----------------------------
# 1) 전처리 유틸 함수들
# -----------------------------
# 이 섹션은 PDF에서 추출한 텍스트를 정리하고 정규화하는 함수들을 포함합니다.

def _normalize_text(t: str) -> str:
    """
    PDF에서 추출한 텍스트를 정규화(정리)합니다.
    
    PDF에서 추출한 텍스트는 다음과 같은 문제들이 있습니다:
    - 하이픈으로 끝나는 단어가 다음 줄로 나뉘어져 있음 (예: "응급-\n처치")
    - 불필요한 줄바꿈이 많음
    - 과도한 공백이 있음
    
    이 함수는 이러한 문제들을 해결하여 임베딩 품질을 향상시킵니다.
    
    Args:
        t: 정규화할 텍스트 문자열
    
    Returns:
        str: 정규화된 텍스트
    """
    # 하이픈 + 줄바꿈 → 단어 결합
    # 정규표현식 r"-\s*\n\s*": 하이픈(-) 뒤에 공백/탭(\s*)이 있고, 줄바꿈(\n)이 있고, 또 공백/탭(\s*)이 있는 패턴
    # 예: "응급-\n처치" → "응급처치"
    t = re.sub(r"-\s*\n\s*", "", t)

    # 줄바꿈을 일괄 공백으로 바꾸면 목록 구조가 무너질 수 있음
    # 따라서: 너무 짧은 줄(예: 한두 단어) 뒤 줄바꿈은 공백으로 완화하는 식의 보수적 규칙 적용
    # 
    # .splitlines(): 텍스트를 줄 단위로 분리
    # [ln.strip() for ln in ...]: 각 줄의 앞뒤 공백 제거
    lines = [ln.strip() for ln in t.splitlines()]
    merged: List[str] = []  # 병합된 줄들을 담을 리스트
    
    for ln in lines:
        if not ln:
            # 빈 줄은 문단 경계로 남겨둠 (문단 구분은 유지해야 함)
            merged.append("")
            continue

        # 이전 줄이 있고, 이전 줄이 너무 짧거나 문장이 끊긴 느낌(마침표/다음표현 없음)이라면 이어붙임
        # 조건:
        # - merged and merged[-1]: 이전 줄이 존재함
        # - len(merged[-1]) < 40: 이전 줄이 40자 미만 (너무 짧음)
        # - not re.search(r"[\.!?…。]$", merged[-1]): 이전 줄이 마침표 등으로 끝나지 않음
        # → 이 경우 문장이 중간에 끊긴 것으로 간주하고 현재 줄과 병합
        if merged and merged[-1] and len(merged[-1]) < 40 and not re.search(r"[\.!?…。]$", merged[-1]):
            merged[-1] = (merged[-1] + " " + ln).strip()
        else:
            merged.append(ln)

    # 다중 공백 정리
    # "\n".join(merged): 병합된 줄들을 다시 줄바꿈으로 연결
    t2 = "\n".join(merged)
    # re.sub(r"[ \t]+", " ", t2): 연속된 공백/탭을 하나의 공백으로 변경
    # .strip(): 앞뒤 공백 제거
    t2 = re.sub(r"[ \t]+", " ", t2).strip()
    return t2


def _find_repeated_lines(pages: List[str], min_ratio: float = 0.6) -> set[str]:
    """
    PDF의 '머리말/꼬리말'처럼 반복적으로 등장하는 라인을 탐지합니다.
    
    PDF 문서에는 보통 각 페이지마다 반복되는 헤더나 푸터가 있습니다.
    이러한 반복 라인은 검색 품질을 떨어뜨리므로 제거해야 합니다.
    
    아이디어:
    - 각 페이지 텍스트를 줄 단위로 분해
    - 특정 라인이 전체 페이지의 일정 비율 이상에서 등장하면 '반복 라인'으로 간주
    
    주의:
    - 너무 공격적으로 지우면 중요한 문장도 사라질 수 있음
    - 그래서 기본값 min_ratio를 비교적 높게(0.6) 잡음
      즉, 전체 페이지의 60% 이상에서 등장하는 라인만 반복 라인으로 간주
    
    Args:
        pages: 각 페이지의 텍스트를 담은 리스트
        min_ratio: 반복 라인으로 간주하기 위한 최소 비율 (0.0 ~ 1.0)
                   기본값 0.6 = 전체 페이지의 60% 이상에서 등장해야 반복 라인으로 간주
    
    Returns:
        set[str]: 반복 라인들의 집합 (제거해야 할 라인들)
    """
    from collections import Counter  # Counter: 각 항목의 개수를 세는 딕셔너리

    total_pages = max(len(pages), 1)  # 전체 페이지 수 (0으로 나누는 것을 방지하기 위해 최소 1)
    cnt = Counter()  # 각 라인이 몇 페이지에서 등장하는지 세는 카운터
    
    for p in pages:
        # 각 페이지의 텍스트를 줄 단위로 분해
        # set([...]): 중복 제거 (같은 페이지 내에서 같은 라인이 여러 번 나와도 1번만 카운트)
        # x.strip(): 앞뒤 공백 제거
        # if x.strip(): 빈 줄 제외
        for ln in set([x.strip() for x in p.splitlines() if x.strip()]):
            # 짧은 라인은 흔히 발생하는 일반 단어일 수 있어 제외 (예: "1", "의", "및" 등)
            if len(ln) < 8:
                continue
            cnt[ln] += 1  # 이 라인이 등장한 페이지 수 증가

    # 반복 라인 찾기
    repeated = set()
    for ln, c in cnt.items():
        # c / total_pages: 이 라인이 등장한 페이지 비율
        # min_ratio 이상이면 반복 라인으로 간주
        if c / total_pages >= min_ratio:
            repeated.add(ln)
    return repeated


# -----------------------------
# 2) PDF → 페이지별 텍스트 추출
# -----------------------------

@dataclass
class PageText:
    """
    한 페이지의 텍스트를 담는 데이터클래스
    
    Attributes:
        page_no: 페이지 번호 (1부터 시작, 사람이 보는 페이지 번호와 일치)
        text: 해당 페이지에서 추출한 텍스트 내용
    """
    page_no: int  # 1-based (사람이 보는 페이지 번호)
    text: str  # 페이지 텍스트


def load_pdf_pages(pdf_path: Path) -> List[PageText]:
    """
    PDF 파일에서 각 페이지의 텍스트를 추출합니다.
    
    PyMuPDF(fitz)를 사용하여 PDF 파일을 열고, 각 페이지에서 텍스트를 추출합니다.
    
    Args:
        pdf_path: PDF 파일의 경로
    
    Returns:
        List[PageText]: 각 페이지의 텍스트를 담은 PageText 객체들의 리스트
                       페이지 번호 순서대로 정렬되어 있습니다 (1페이지, 2페이지, ...)
    """
    # fitz.open(): PyMuPDF를 사용하여 PDF 파일 열기
    doc = fitz.open(pdf_path)
    pages: List[PageText] = []  # 결과를 담을 리스트
    
    # doc.page_count: PDF의 총 페이지 수
    for i in range(doc.page_count):
        # doc.load_page(i): i번째 페이지 로드 (0부터 시작, 즉 첫 페이지는 0)
        page = doc.load_page(i)
        # page.get_text("text"): 페이지에서 텍스트 추출 (가장 단순한 텍스트 모드)
        raw = page.get_text("text")
        # PageText 객체 생성 (페이지 번호는 1부터 시작하므로 i + 1)
        pages.append(PageText(page_no=i + 1, text=raw))
    
    return pages


# -----------------------------
# 3) 청킹 전략
# -----------------------------

_SECTION_HINT = re.compile(
    r"^(?:[IVX]+\.|\d+\.|\d+\)|[가-힣]\)|\(?[0-9]{1,2}\)?\s)"
)

def split_into_paragraphs(page_text: str) -> List[str]:
    """
    페이지 텍스트를 '문단' 단위로 분리합니다.
    
    문단은 보통 빈 줄로 구분되어 있습니다.
    이 함수는 빈 줄을 기준으로 텍스트를 문단 단위로 나눕니다.
    
    Args:
        page_text: 페이지 텍스트
    
    Returns:
        List[str]: 문단들로 분리된 텍스트 리스트
    """
    # re.split(r"\n\s*\n", page_text): 줄바꿈 + 공백 + 줄바꿈 패턴을 기준으로 분리 (빈 줄 기준)
    # [p.strip() for p in ... if p.strip()]: 각 부분의 앞뒤 공백 제거하고, 빈 부분 제외
    parts = [p.strip() for p in re.split(r"\n\s*\n", page_text) if p.strip()]
    return parts


def chunk_paragraphs(
    paragraphs: List[Tuple[int, str]],
    max_chars: int = 1600,
    overlap_chars: int = 200,
) -> List[Dict]:
    """
    문단들을 누적하여 chunk(문서 조각)를 만듭니다.
    
    이 함수는 여러 문단을 합쳐서 하나의 chunk로 만듭니다.
    각 chunk는 최대 max_chars 길이를 넘지 않도록 합니다.
    
    왜 chunk로 나누나요?
    - 전체 문서를 한 번에 검색하면 정확도가 떨어집니다
    - 작은 조각으로 나누면 질문과 정확히 관련된 부분만 찾을 수 있습니다
    - 각 chunk는 최대 길이 제한이 있어 AI 모델의 입력 제한에 맞출 수 있습니다
    
    Args:
        paragraphs: (페이지 번호, 문단 텍스트) 튜플의 리스트
        max_chars: 하나의 chunk가 가질 수 있는 최대 문자 수 (기본값: 1600)
        overlap_chars: chunk 간 오버랩(겹침) 문자 수 (현재는 최소 적용, 기본값: 200)
    
    Returns:
        List[Dict]: chunk 딕셔너리들의 리스트
                   각 딕셔너리는 {"text": "...", "page_start": 1, "page_end": 2} 형태
    """
    chunks: List[Dict] = []  # 결과 chunk들을 담을 리스트
    buf: List[Tuple[int, str]] = []  # 현재 chunk를 만들고 있는 문단들의 버퍼
    buf_len = 0  # 현재 버퍼의 총 문자 수

    def flush():
        """
        현재 버퍼의 내용을 하나의 chunk로 만들어 chunks 리스트에 추가합니다.
        
        이 함수는 버퍼가 가득 찼을 때 또는 모든 문단을 처리했을 때 호출됩니다.
        """
        nonlocal buf, buf_len  # 외부 스코프의 변수를 수정하기 위해 nonlocal 사용
        if not buf:
            return  # 버퍼가 비어있으면 아무것도 하지 않음
        
        # 버퍼에 있는 모든 문단에서 페이지 번호 추출
        pages = [p for p, _ in buf]
        # 버퍼에 있는 모든 문단의 텍스트를 빈 줄 2개로 구분하여 합침
        text = "\n\n".join([t for _, t in buf]).strip()
        
        # chunk 딕셔너리 생성
        chunks.append({
            "text": text,  # chunk의 텍스트 내용
            "page_start": min(pages),  # 이 chunk가 포함하는 페이지 중 가장 작은 번호
            "page_end": max(pages),  # 이 chunk가 포함하는 페이지 중 가장 큰 번호
        })
        
        # 버퍼 초기화
        buf = []
        buf_len = 0

    # 각 문단을 순회하면서 chunk 생성
    for page_no, para in paragraphs:
        para = para.strip()  # 문단의 앞뒤 공백 제거
        if not para:
            continue  # 빈 문단은 건너뜀

        # 문단 하나가 너무 길면 강제로 자름 (문단 내부에서 문장/줄 기준)
        if len(para) > max_chars:
            # 줄 단위로 쪼개서 다시 누적
            lines = [ln.strip() for ln in para.splitlines() if ln.strip()]
            for ln in lines:
                # 현재 줄을 추가하면 max_chars를 넘으면 flush하고 새 chunk 시작
                if buf_len + len(ln) + 2 > max_chars:  # +2는 줄바꿈 문자 고려
                    flush()
                buf.append((page_no, ln))
                buf_len += len(ln) + 2
            continue  # 다음 문단으로 넘어감

        # 현재 문단을 추가하면 max_chars를 넘으면 flush하고 새 chunk 시작
        if buf_len + len(para) + 4 > max_chars:  # +4는 문단 구분 문자 고려
            flush()

            # overlap: 직전 chunk의 끝부분 일부를 가져오려면,
            # 실제 구현이 복잡해질 수 있어 "간단 버전"으로는 생략하거나 최소화.
            # 여기서는 flush 직전 내용을 저장해 overlap을 구성합니다.
            # (정확한 overlap이 필요하면 이후에 개선 가능)
        
        # 현재 문단을 버퍼에 추가
        buf.append((page_no, para))
        buf_len += len(para) + 4  # +4는 문단 구분 문자 고려

    # 마지막 버퍼의 내용도 chunk로 만들어서 추가
    flush()
    return chunks


# -----------------------------
# 4) 임베딩 + FAISS 저장
# -----------------------------

def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 32) -> np.ndarray:
    """
    텍스트 리스트를 임베딩 벡터로 변환합니다.
    
    이 함수는 여러 텍스트를 한 번에 임베딩하여 numpy 배열로 반환합니다.
    배치 처리를 통해 효율적으로 임베딩을 생성합니다.
    
    참고:
    - E5 계열 모델은 보통 'query:'/'passage:' 프리픽스를 권장하지만,
      여기서는 단순화를 위해 프리픽스를 사용하지 않습니다.
      (개선 시: 질문 임베딩에는 'query:' prefix를 사용할 수 있습니다)
    
    Args:
        model: SentenceTransformer 모델 객체 (임베딩 생성용)
        texts: 임베딩할 텍스트들의 리스트
        batch_size: 한 번에 처리할 텍스트의 개수 (기본값: 32)
                   메모리가 부족하면 이 값을 줄이면 됩니다
    
    Returns:
        np.ndarray: 임베딩 벡터들의 배열 (shape: [텍스트 개수, 벡터 차원])
                   dtype은 float32입니다
    """
    # model.encode(): SentenceTransformer 모델을 사용하여 텍스트를 벡터로 변환
    embs = model.encode(
        texts,  # 임베딩할 텍스트 리스트
        batch_size=batch_size,  # 배치 크기 (한 번에 처리할 개수)
        show_progress_bar=True,  # 진행 상황 표시 (tqdm 진행 바)
        normalize_embeddings=True,  # 벡터 정규화 (길이를 1로 만듦)
        # 코사인 유사도 계산을 위해 정규화가 필요합니다
        # 정규화된 벡터의 내적(inner product)은 코사인 유사도와 동일합니다
    )
    # numpy 배열로 변환 (FAISS가 numpy 배열을 요구함)
    return np.asarray(embs, dtype=np.float32)


def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    """
    임베딩 벡터들로부터 FAISS 검색 인덱스를 생성합니다.
    
    이 함수는 정규화된 벡터들(코사인 유사도 검색용)을 FAISS 인덱스로 변환합니다.
    IndexFlatIP (Inner Product)를 사용하여 코사인 유사도 검색을 수행할 수 있습니다.
    
    Args:
        vectors: 임베딩 벡터들의 배열 (shape: [벡터 개수, 벡터 차원])
                정규화되어 있어야 합니다 (normalize_embeddings=True로 생성된 벡터)
    
    Returns:
        faiss.Index: FAISS 검색 인덱스 객체
    """
    # vectors.shape[1]: 벡터의 차원 수 (예: 768)
    dim = vectors.shape[1]
    
    # faiss.IndexFlatIP: Inner Product (내적) 기반 검색 인덱스
    # 정규화된 벡터의 내적은 코사인 유사도와 동일합니다
    index = faiss.IndexFlatIP(dim)
    
    # index.add(): 벡터들을 인덱스에 추가
    index.add(vectors)
    
    return index


# -----------------------------
# 5) CLI 엔트리포인트
# -----------------------------

def main():
    """
    메인 함수: PDF를 읽어서 FAISS 인덱스를 생성하는 전체 프로세스를 실행합니다.
    
    이 함수는 다음 단계를 수행합니다:
    1. 명령줄 인자 파싱
    2. PDF 파일에서 텍스트 추출
    3. 텍스트 정규화 및 전처리
    4. 텍스트를 작은 조각(chunk)으로 나누기
    5. 각 chunk를 임베딩 벡터로 변환
    6. FAISS 인덱스 생성 및 저장
    7. 메타데이터 및 chunk 데이터 저장
    
    사용 예:
        python -m ingest.build_index --pdf data/guide_2023.pdf --out artifacts
    """
    # ========================================
    # 명령줄 인자 파싱
    # ========================================
    ap = argparse.ArgumentParser(description="PDF를 FAISS 인덱스로 변환")
    ap.add_argument("--pdf", type=str, required=True, help="PDF 파일 경로")
    ap.add_argument("--out", type=str, required=True, help="산출물 폴더(artifacts)")
    ap.add_argument("--max_chars", type=int, default=1600, help="chunk 최대 문자 수(대략 토큰 제한 역할)")
    ap.add_argument("--overlap_chars", type=int, default=200, help="chunk 오버랩 문자 수(현재는 최소 적용)")
    args = ap.parse_args()  # 명령줄 인자 파싱

    # 경로 설정
    pdf_path = Path(args.pdf)  # PDF 파일 경로
    out_dir = Path(args.out)  # 출력 디렉토리 경로
    # out_dir.mkdir(): 디렉토리 생성 (없으면 생성, 있으면 그대로)
    # parents=True: 상위 디렉토리도 자동 생성
    # exist_ok=True: 이미 존재해도 에러 발생하지 않음
    out_dir.mkdir(parents=True, exist_ok=True)

    # ========================================
    # 1단계: PDF 파일에서 텍스트 추출
    # ========================================
    print("[1/7] PDF 파일 로딩 중...")
    pages = load_pdf_pages(pdf_path)  # 각 페이지의 텍스트를 추출

    # ========================================
    # 2단계: 반복 라인 탐지 (헤더/푸터 제거용)
    # ========================================
    print("[2/7] 반복 라인 탐지 중...")
    raw_pages = [p.text for p in pages]  # 페이지 텍스트만 추출
    repeated = _find_repeated_lines(raw_pages, min_ratio=0.6)  # 반복 라인 찾기

    # ========================================
    # 3단계: 페이지별 정규화 + 반복 라인 제거 + 문단 분리
    # ========================================
    print("[3/7] 텍스트 정규화 및 전처리 중...")
    paras: List[Tuple[int, str]] = []  # (페이지 번호, 문단 텍스트) 튜플 리스트
    
    for p in pages:
        t = p.text  # 페이지 텍스트
        
        # 반복 라인 제거 (정확한 매칭 기반)
        lines = []
        for ln in t.splitlines():  # 각 줄을 순회
            ln2 = ln.strip()  # 앞뒤 공백 제거
            if ln2 in repeated:  # 반복 라인이면 건너뜀
                continue
            lines.append(ln)  # 반복 라인이 아니면 추가
        
        t = "\n".join(lines)  # 줄들을 다시 합침
        t = _normalize_text(t)  # 텍스트 정규화 (하이픈 결합, 공백 정리 등)

        # 문단으로 분리
        for para in split_into_paragraphs(t):
            paras.append((p.page_no, para))  # (페이지 번호, 문단) 튜플 추가

    # ========================================
    # 4단계: 청킹 (문단들을 문서 조각으로 나누기)
    # ========================================
    print("[4/7] 문서 조각(chunk) 생성 중...")
    chunks = chunk_paragraphs(paras, max_chars=args.max_chars, overlap_chars=args.overlap_chars)

    # ========================================
    # chunk_id 부여 + 소스 정보 저장
    # ========================================
    print("[5/7] 메타데이터 준비 중...")
    meta: List[Dict] = []  # chunk 메타데이터 리스트
    chunk_texts: List[str] = []  # chunk 텍스트만 담은 리스트 (임베딩용)
    
    for i, ch in enumerate(chunks):
        chunk_id = f"chunk_{i:06d}"  # 고유 ID 생성 (예: "chunk_000001")
        meta.append({
            "chunk_id": chunk_id,
            "page_start": ch["page_start"],  # 시작 페이지
            "page_end": ch["page_end"],  # 끝 페이지
            "source": pdf_path.name,  # PDF 파일명
            # section_path는 고급 파싱(목차/헤더 인식) 단계에서 추가하는 것이 이상적
            # 현재는 자동으로 "(auto)"로 설정
            "section_path": "(auto)",
        })
        chunk_texts.append(ch["text"])  # 텍스트만 따로 저장 (임베딩 생성용)

    # ========================================
    # 5단계: 임베딩 생성 (텍스트를 벡터로 변환)
    # ========================================
    print("[6/7] 임베딩 생성 중... (시간이 걸릴 수 있습니다)")
    # SentenceTransformer 모델 로드 (설정에서 지정한 모델)
    model = SentenceTransformer(SETTINGS.embedding_model_name)
    # 모든 chunk 텍스트를 임베딩 벡터로 변환
    vectors = embed_texts(model, chunk_texts)

    # ========================================
    # 6단계: FAISS 인덱스 생성 및 저장
    # ========================================
    print("[7/7] FAISS 인덱스 생성 및 저장 중...")
    index = build_faiss_index(vectors)  # FAISS 인덱스 생성
    # 인덱스를 파일로 저장
    faiss.write_index(index, str(out_dir / "faiss.index"))

    # ========================================
    # 7단계: 메타데이터 및 chunk 데이터 저장
    # ========================================
    # meta.json 저장 (메타데이터만, 텍스트 제외)
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),  # JSON 형식으로 변환 (들여쓰기 2칸)
        encoding="utf-8"
    )

    # chunks.jsonl 저장 (텍스트 포함, JSON Lines 형식)
    # JSON Lines: 각 줄이 하나의 JSON 객체 (스트리밍 처리에 유리)
    with (out_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for m, txt in zip(meta, chunk_texts):
            # 메타데이터와 텍스트를 합쳐서 하나의 레코드로 만들기
            rec = {**m, "text": txt}  # 딕셔너리 병합 (Python 3.5+)
            # JSON 문자열로 변환 후 파일에 쓰기
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("[OK] Index built:", out_dir)
    print(f"  - {len(chunks)} chunks created")
    print(f"  - Index file: {out_dir / 'faiss.index'}")
    print(f"  - Metadata file: {out_dir / 'meta.json'}")
    print(f"  - Chunks file: {out_dir / 'chunks.jsonl'}")


if __name__ == "__main__":
    main()
