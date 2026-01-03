"""
rag/formatting.py

이 파일은 검색 결과(문서 조각들)를 사용자에게 보여주기 좋은 형식으로 변환하는 함수들을 담고 있습니다.
Streamlit UI에서 근거(출처)를 예쁘게 표시하기 위한 포맷터입니다.
"""

from __future__ import annotations  # 미래 버전 Python 호환성

from typing import List, Dict  # 타입 힌팅용
from dataclasses import asdict  # 데이터클래스 객체를 딕셔너리로 변환하는 함수

from .retriever import RetrievedChunk  # 검색 결과를 나타내는 데이터클래스


def citations_markdown(chunks: List[RetrievedChunk]) -> str:
    """
    검색된 문서 조각들을 마크다운 형식의 문자열로 변환합니다.
    
    이 함수는 검색 결과를 사용자가 읽기 쉬운 형식으로 변환합니다.
    예: "- 근거 1: guide_2023.pdf p.10-12 / score=0.856 / (auto)"
    
    Args:
        chunks: 검색된 문서 조각들의 리스트 (RetrievedChunk 객체들)
                유사도 점수가 높은 순서로 정렬된 리스트입니다
    
    Returns:
        str: 마크다운 형식의 문자열 (각 줄이 하나의 근거를 나타냄)
             예:
             - 근거 1: guide_2023.pdf p.10-12 / score=0.856 / (auto)
             - 근거 2: guide_2023.pdf p.15-17 / score=0.782 / (auto)
    """
    lines = []  # 결과 문자열들을 담을 리스트
    
    # enumerate(chunks, start=1): 리스트의 각 요소와 인덱스를 가져옴 (인덱스는 1부터 시작)
    # i: 근거 번호 (1, 2, 3, ...)
    # ch: RetrievedChunk 객체 (검색된 문서 조각 하나)
    for i, ch in enumerate(chunks, start=1):
        # f-string을 사용하여 문자열 포맷팅
        # - 근거 {i}: 근거 번호
        # {ch.source}: 문서 파일명 (예: "guide_2023.pdf")
        # p.{ch.page_start}-{ch.page_end}: 페이지 범위 (예: "p.10-12")
        # score={ch.score:.3f}: 유사도 점수 (소수점 3자리까지 표시, 예: "score=0.856")
        # {ch.section_path}: 섹션 경로 (현재는 "(auto)"로 고정)
        lines.append(f"- 근거 {i}: {ch.source} p.{ch.page_start}-{ch.page_end} / score={ch.score:.3f} / {ch.section_path}")
    
    # "\n".join(lines): 리스트의 각 요소를 줄바꿈 문자(\n)로 연결하여 하나의 문자열로 만듦
    return "\n".join(lines)


def chunks_as_dicts(chunks: List[RetrievedChunk]) -> List[Dict]:
    """
    RetrievedChunk 객체들의 리스트를 딕셔너리 리스트로 변환합니다.
    
    이 함수는 데이터를 JSON 등으로 저장하거나 전송할 때 유용합니다.
    데이터클래스 객체는 직접 JSON으로 변환할 수 없지만, 딕셔너리는 변환 가능합니다.
    
    Args:
        chunks: RetrievedChunk 객체들의 리스트
    
    Returns:
        List[Dict]: 각 RetrievedChunk 객체가 딕셔너리로 변환된 리스트
                    예: [{"chunk_id": "chunk_000001", "score": 0.856, ...}, ...]
    
    참고:
        asdict(): dataclasses 모듈의 함수로, 데이터클래스 객체를 딕셔너리로 변환합니다
    """
    # 리스트 컴프리헨션: 각 chunk 객체를 asdict()로 딕셔너리로 변환
    return [asdict(c) for c in chunks]
