"""
rag/retriever.py

이 파일은 FAISS 인덱스를 사용하여 문서에서 질문과 관련된 내용을 검색하는 기능을 제공합니다.

주요 기능:
- FAISS 인덱스 로드
- 사용자 질문을 임베딩(숫자로 변환)
- FAISS 인덱스에서 유사한 문서 조각 검색
- 검색 결과를 RetrievedChunk 객체로 반환

중요 포인트:
- 인덱스는 normalize_embeddings=True로 생성했기 때문에,
  검색 점수는 코사인 유사도와 동일한 의미를 가집니다(정확히는 inner product == cosine).
  즉, 점수가 높을수록 질문과 더 유사한 내용입니다.
"""

from __future__ import annotations  # 미래 버전 Python 호환성

import json  # JSON 파일 읽기/쓰기용
from dataclasses import dataclass  # 데이터클래스 정의용
from pathlib import Path  # 파일 경로 다루기용
from typing import List, Dict, Tuple  # 타입 힌팅용

import faiss  # Facebook의 빠른 검색 라이브러리
import numpy as np  # 수치 계산용 라이브러리 (벡터 연산 등)
from sentence_transformers import SentenceTransformer  # 텍스트를 벡터(임베딩)로 변환하는 모델

from config import SETTINGS  # 프로젝트 설정 (임베딩 모델 이름 등)


@dataclass
class RetrievedChunk:
    """
    검색된 문서 조각(청크)을 나타내는 데이터클래스
    
    검색 결과로 반환되는 각 문서 조각의 정보를 담고 있습니다.
    
    Attributes:
        chunk_id: 청크의 고유 ID (예: "chunk_000001")
        score: 유사도 점수 (0.0 ~ 1.0에 가까운 값, 높을수록 유사함)
        page_start: 시작 페이지 번호 (1부터 시작)
        page_end: 끝 페이지 번호
        section_path: 섹션 경로 (현재는 "(auto)"로 고정)
        source: 문서 파일명 (예: "guide_2023.pdf")
        text: 문서 조각의 실제 텍스트 내용
    """
    chunk_id: str  # 청크 고유 ID
    score: float  # 유사도 점수
    page_start: int  # 시작 페이지
    page_end: int  # 끝 페이지
    section_path: str  # 섹션 경로
    source: str  # 문서 파일명
    text: str  # 텍스트 내용


class FaissRetriever:
    """
    FAISS 인덱스를 사용하여 문서를 검색하는 클래스
    
    이 클래스는 다음을 수행합니다:
    1. FAISS 인덱스 파일 로드
    2. 문서 조각(chunks) 메타데이터 로드
    3. 사용자 질문을 임베딩으로 변환
    4. FAISS 인덱스에서 유사한 문서 조각 검색
    """
    
    def __init__(self, artifacts_dir: Path, embedding_model_name: str = SETTINGS.embedding_model_name):
        """
        FaissRetriever 객체를 초기화합니다.
        
        초기화 시 다음 작업을 수행합니다:
        - 임베딩 모델 로드 (질문을 벡터로 변환하기 위해 필요)
        - FAISS 인덱스 파일 로드 (검색을 위해 필요)
        - 문서 조각(chunks) 메타데이터 로드 (검색 결과에 텍스트와 메타데이터를 포함하기 위해 필요)
        
        Args:
            artifacts_dir: 인덱스 파일들이 저장된 디렉토리 경로
                          이 디렉토리에는 다음 파일들이 있어야 합니다:
                          - faiss.index: FAISS 인덱스 파일
                          - chunks.jsonl: 문서 조각들의 텍스트와 메타데이터
            embedding_model_name: 사용할 임베딩 모델 이름
                                 기본값은 config.py의 SETTINGS에서 가져옵니다
                                 예: "intfloat/multilingual-e5-base"
        """
        self.artifacts_dir = artifacts_dir  # 인덱스 파일 디렉토리 경로 저장
        
        # SentenceTransformer 모델 로드
        # 이 모델은 텍스트를 벡터(숫자 배열)로 변환하는 데 사용됩니다
        # 처음 실행 시 Hugging Face에서 모델을 다운로드할 수 있습니다
        self.model = SentenceTransformer(embedding_model_name)

        # ========================================
        # FAISS 인덱스 파일 로드
        # ========================================
        # faiss.read_index(): FAISS 인덱스 파일을 메모리로 읽어옵니다
        # str(artifacts_dir / "faiss.index"): 경로를 문자열로 변환 (FAISS가 문자열 경로를 요구함)
        # 이 인덱스에는 모든 문서 조각의 임베딩 벡터가 저장되어 있습니다
        self.index = faiss.read_index(str(artifacts_dir / "faiss.index"))

        # ========================================
        # 문서 조각(chunks) 메타데이터 로드
        # ========================================
        # chunks.jsonl은 JSON Lines 형식의 파일입니다
        # 각 줄이 하나의 JSON 객체이며, 문서 조각의 텍스트와 메타데이터를 담고 있습니다
        self.records: List[Dict] = []  # 문서 조각 정보를 담을 리스트
        chunks_path = artifacts_dir / "chunks.jsonl"  # chunks.jsonl 파일 경로
        
        # 파일을 읽어서 각 줄을 JSON으로 파싱하여 리스트에 추가
        with chunks_path.open("r", encoding="utf-8") as f:  # 파일 열기 (UTF-8 인코딩)
            for line in f:  # 파일의 각 줄을 순회
                # json.loads(line): JSON 문자열을 딕셔너리로 변환
                # 예: '{"chunk_id": "chunk_000001", "text": "...", ...}' → {"chunk_id": "chunk_000001", ...}
                self.records.append(json.loads(line))

    def _embed_query(self, query: str) -> np.ndarray:
        """
        사용자 질문을 임베딩 벡터로 변환합니다 (내부 메서드)
        
        이 메서드는 질문 텍스트를 컴퓨터가 이해할 수 있는 숫자 배열(벡터)로 변환합니다.
        이 벡터를 사용하여 FAISS 인덱스에서 유사한 문서 조각을 찾을 수 있습니다.
        
        Args:
            query: 사용자가 입력한 질문 텍스트
        
        Returns:
            np.ndarray: 질문을 임베딩한 벡터 (numpy 배열, float32 타입)
        """
        # E5 계열 모델 권장 사항: "query: " 접두사를 붙이면 검색 성능이 좋아집니다
        # E5 모델은 "query: " 접두사가 있으면 이것이 검색 질문임을 이해하고 더 나은 임베딩을 생성합니다
        # .strip(): 앞뒤 공백 제거
        q = "query: " + query.strip()
        
        # self.model.encode(): SentenceTransformer 모델을 사용하여 텍스트를 벡터로 변환
        # [q]: 리스트 형태로 전달 (모델은 리스트를 입력으로 받음)
        # normalize_embeddings=True: 벡터를 정규화 (길이를 1로 만듦)
        #   정규화된 벡터의 내적(inner product)은 코사인 유사도와 동일한 의미를 가집니다
        #   이것이 FAISS 검색 점수가 코사인 유사도를 나타내는 이유입니다
        emb = self.model.encode([q], normalize_embeddings=True)
        
        # numpy 배열로 변환 (FAISS가 numpy 배열을 요구함)
        # dtype=np.float32: 32비트 부동소수점 타입 (메모리 효율적이고 FAISS 표준)
        return np.asarray(emb, dtype=np.float32)

    def search(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        """
        질문과 유사한 문서 조각들을 검색합니다.
        
        이 메서드는 RAG 시스템의 핵심 기능입니다.
        사용자 질문과 관련된 문서 조각들을 찾아서 반환합니다.
        
        검색 과정:
        1. 질문을 임베딩 벡터로 변환
        2. FAISS 인덱스에서 유사도 점수가 높은 상위 k개 검색
        3. 검색 결과를 RetrievedChunk 객체 리스트로 변환하여 반환
        
        Args:
            query: 사용자가 입력한 질문 텍스트
            top_k: 검색할 문서 조각의 개수 (기본값: 5)
                   예: top_k=5 이면 유사도 점수가 높은 5개를 가져옵니다
        
        Returns:
            List[RetrievedChunk]: 검색된 문서 조각들의 리스트
                                유사도 점수가 높은 순서로 정렬되어 있습니다
                                (첫 번째 요소가 가장 유사한 문서 조각)
        """
        # 1단계: 질문을 임베딩 벡터로 변환
        qv = self._embed_query(query)
        
        # 2단계: FAISS 인덱스에서 검색
        # self.index.search(qv, top_k): FAISS 인덱스에서 가장 유사한 top_k개를 찾습니다
        # qv: 질문의 임베딩 벡터
        # top_k: 가져올 개수
        # 
        # 반환값:
        # - scores: 유사도 점수 배열 (shape: [1, top_k])
        # - idxs: 검색된 문서 조각의 인덱스 배열 (shape: [1, top_k])
        #   이 인덱스는 self.records 리스트의 인덱스와 일치합니다
        scores, idxs = self.index.search(qv, top_k)

        # 3단계: 검색 결과를 RetrievedChunk 객체 리스트로 변환
        out: List[RetrievedChunk] = []  # 결과를 담을 리스트
        
        # zip(scores[0].tolist(), idxs[0].tolist()): 점수와 인덱스를 짝지어 순회
        # scores[0]: 첫 번째(그리고 유일한) 쿼리의 점수 배열
        # .tolist(): numpy 배열을 Python 리스트로 변환
        for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
            # idx < 0: FAISS가 검색 결과를 찾지 못한 경우 (일반적으로 발생하지 않음)
            if idx < 0:
                continue  # 이 결과는 건너뜀
            
            # self.records[idx]: 인덱스에 해당하는 문서 조각의 메타데이터와 텍스트
            rec = self.records[idx]
            
            # RetrievedChunk 객체 생성하여 리스트에 추가
            out.append(RetrievedChunk(
                chunk_id=rec["chunk_id"],  # 청크 ID
                score=float(score),  # 유사도 점수 (코사인 유사도)
                page_start=int(rec["page_start"]),  # 시작 페이지
                page_end=int(rec["page_end"]),  # 끝 페이지
                section_path=rec.get("section_path", "(auto)"),  # 섹션 경로 (없으면 "(auto)")
                source=rec.get("source", ""),  # 문서 파일명 (없으면 빈 문자열)
                text=rec["text"],  # 문서 조각의 텍스트 내용
            ))
        
        return out  # 검색 결과 리스트 반환
