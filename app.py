"""
app.py

이 파일은 Streamlit으로 만든 웹 기반 챗봇 인터페이스입니다.
사용자가 질문을 입력하면, PDF 문서에서 관련 정보를 찾아 답변을 생성합니다.

주요 특징:
- 임베딩/검색: 모두 로컬에서 실행 (인터넷 불필요)
- LLM(답변 생성): Ollama를 통해 로컬에서 실행

실행 전 준비사항:
1) PDF 인덱싱: python -m ingest.build_index --pdf data/guide_2023.pdf --out artifacts
2) Ollama 모델 다운로드: ollama pull llama3.1:8b-instruct
3) 실행: streamlit run app.py
"""

from __future__ import annotations  # 미래 버전 Python 호환성을 위한 임포트 (타입 힌팅 관련)

from pathlib import Path  # 파일 경로를 다루기 위한 모듈
from typing import List  # 타입 힌팅용 (List 타입 명시)

import streamlit as st  # 웹 인터페이스를 만들기 위한 라이브러리

# 프로젝트 내부 모듈들 임포트
from config import SETTINGS  # 설정값 (모델 이름, 기본값 등)
from rag.retriever import FaissRetriever, RetrievedChunk  # 문서 검색 기능
from rag.ollama_client import OllamaClient  # Ollama AI 모델과 통신
from rag.prompting import build_prompt  # AI에게 보낼 프롬프트 생성
from rag.formatting import citations_markdown  # 검색 결과를 마크다운 형식으로 변환

# 인덱스 파일들이 저장된 디렉토리 경로
# artifacts 폴더에는 faiss.index, meta.json, chunks.jsonl 파일이 저장됩니다
ARTIFACTS_DIR = Path("artifacts")


# -----------------------------
# Streamlit 캐시 함수 (무거운 리소스 로딩 최적화)
# -----------------------------

@st.cache_resource
def load_retriever() -> FaissRetriever:
    """
    문서 검색기(FaissRetriever)를 로드하는 함수
    
    @st.cache_resource 데코레이터의 역할:
    - 이 함수는 처음 호출될 때만 실행되고, 그 결과를 메모리에 캐시(저장)합니다
    - 이후 호출에서는 캐시된 결과를 재사용합니다
    - 이유: FAISS 인덱스와 임베딩 모델을 로드하는 것은 시간이 오래 걸리기 때문입니다
    - 사용자가 질문할 때마다 다시 로드하면 매우 느려집니다
    
    Returns:
        FaissRetriever: 문서 검색 기능을 제공하는 객체
    """
    # ARTIFACTS_DIR에서 FAISS 인덱스를 로드하고, 임베딩 모델도 함께 로드합니다
    return FaissRetriever(ARTIFACTS_DIR, embedding_model_name=SETTINGS.embedding_model_name)


@st.cache_resource
def load_ollama(model_name: str) -> OllamaClient:
    """
    Ollama 클라이언트를 로드하는 함수
    
    @st.cache_resource 데코레이터의 역할:
    - 마찬가지로 처음 한 번만 실행되고 결과를 캐시합니다
    - OllamaClient 객체를 생성하는 것은 가볍지만, 일관성을 위해 캐시를 사용합니다
    
    Args:
        model_name: 사용할 Ollama 모델 이름 (예: "llama3.1:8b-instruct")
    
    Returns:
        OllamaClient: Ollama 서버와 통신하는 클라이언트 객체
    """
    return OllamaClient(model=model_name)


# -----------------------------
# UI (사용자 인터페이스) 설정
# -----------------------------

# Streamlit 페이지 설정
# page_title: 브라우저 탭에 표시되는 제목
# layout="wide": 레이아웃을 넓게 설정 (더 많은 공간 활용)
st.set_page_config(page_title="119 현장응급처치 RAG", layout="wide")

# 페이지 제목 표시 (가장 큰 제목)
st.title("119 구급대원 현장응급처치 RAG 챗봇 (로컬)")

# 사이드바 생성 (왼쪽에 표시되는 설정 패널)
with st.sidebar:
    st.header("설정")  # 사이드바 내부 제목
    
    # Top-k 슬라이더: 검색할 문서 조각 개수를 설정
    # min_value=2: 최소 2개 (너무 적으면 정보 부족)
    # max_value=8: 최대 8개 (너무 많으면 생성 시간 증가)
    # value=SETTINGS.top_k: 기본값은 config.py에서 설정한 값 (3, 속도 최적화)
    # step=1: 1개 단위로 조정
    # 주의: 값을 늘리면 더 많은 정보를 참고하지만 답변 생성 시간이 늘어납니다
    top_k = st.slider("Top-k(근거 조각 개수)", min_value=2, max_value=8, value=SETTINGS.top_k, step=1, 
                     help="값이 클수록 정확하지만 느립니다. 빠른 응답을 원하면 2-3을 권장합니다.")
    
    # 최소 유사도 슬라이더: 검색 결과의 유사도가 이 값보다 낮으면 "추가 질문" 모드로 전환
    # min_value=0.10: 최소 0.10 (10% 유사도)
    # max_value=0.60: 최대 0.60 (60% 유사도)
    # value=SETTINGS.min_similarity: 기본값은 config.py에서 설정한 값 (0.30)
    # step=0.01: 0.01 단위로 조정
    min_sim = st.slider("최소 유사도(이하면 되묻기)", min_value=0.10, max_value=0.60, value=SETTINGS.min_similarity, step=0.01)
    
    # 체크박스: 검색한 문서 조각 정보를 화면에 표시할지 여부
    # value=True: 기본값은 표시
    show_sources = st.checkbox("근거(페이지/조각) 표시", value=True)
    
    # 텍스트 입력: 사용할 Ollama 모델 이름을 입력
    # value=SETTINGS.ollama_model: 기본값은 config.py에서 설정한 값
    ollama_model = st.text_input("Ollama 모델명", value=SETTINGS.ollama_model)
    
    # Temperature 슬라이더: AI 답변의 창의성/랜덤성을 조절
    # 0.0에 가까울수록: 일관적이고 예측 가능한 답변 (같은 질문에 같은 답변)
    # 1.0에 가까울수록: 다양하고 창의적인 답변 (같은 질문에 다른 답변 가능)
    # 의료/응급처치 같은 정확성이 중요한 분야에서는 낮은 값(0.2)을 권장
    temperature = st.slider("LLM temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

    st.divider()  # 구분선 표시
    st.caption("인덱스 파일이 없으면 먼저 build_index를 실행하세요.")  # 작은 글씨로 안내 문구


# -----------------------------
# 인덱스 파일 존재 여부 확인
# -----------------------------
# FAISS 인덱스 파일이 없으면 검색을 할 수 없으므로, 먼저 인덱싱을 해야 합니다
if not (ARTIFACTS_DIR / "faiss.index").exists():
    # 에러 메시지 표시 (빨간색 박스로 표시됨)
    st.error("artifacts/faiss.index 가 없습니다. 먼저 인덱스를 생성하세요.")
    # 인덱스 생성 명령어를 코드 블록으로 표시
    st.code("python -m ingest.build_index --pdf data/guide_2023.pdf --out artifacts", language="bash")
    st.stop()  # 프로그램 실행 중단 (이 아래 코드는 실행되지 않음)

# -----------------------------
# 검색기와 Ollama 클라이언트 로드
# -----------------------------
# @st.cache_resource로 캐시된 함수들을 호출하여 객체를 가져옵니다
# 처음 호출 시에는 로드에 시간이 걸리지만, 이후에는 캐시된 결과를 사용하므로 빠릅니다
retriever = load_retriever()  # 문서 검색기 로드
ollama = load_ollama(ollama_model)  # Ollama 클라이언트 로드

# -----------------------------
# 대화 기록 초기화
# -----------------------------
# st.session_state: Streamlit의 세션 상태 (페이지 새로고침 전까지 유지되는 데이터)
# "messages" 키가 없으면 빈 리스트로 초기화
# 각 메시지는 딕셔너리 형태: {"role": "user" 또는 "assistant", "content": "메시지 내용", "sources_md": "출처 정보(선택)"}
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{role, content, sources?}]

# -----------------------------
# 기존 대화 기록을 화면에 표시
# -----------------------------
# 사용자가 이전에 나눈 대화를 화면에 다시 그려줍니다
# 페이지를 새로고침하면 session_state가 초기화되므로 대화가 사라집니다
for m in st.session_state.messages:
    # st.chat_message: 채팅 메시지 스타일로 표시 (role에 따라 왼쪽/오른쪽 배치)
    with st.chat_message(m["role"]):  # "user" 또는 "assistant"
        st.markdown(m["content"])  # 메시지 내용을 마크다운 형식으로 표시
        
        # "근거 보기" 체크박스가 체크되어 있고, 출처 정보가 있으면 표시
        if show_sources and m.get("sources_md"):
            # st.expander: 접을 수 있는 섹션 (기본적으로 접혀있고, 클릭하면 펼쳐짐)
            with st.expander("근거 보기"):
                st.markdown(m["sources_md"])  # 출처 정보 표시

# -----------------------------
# 사용자 입력 받기
# -----------------------------
# st.chat_input: 채팅 입력 필드 (하단에 표시되는 입력창)
# 사용자가 Enter를 누르거나 전송 버튼을 누르면 입력된 텍스트가 user_q 변수에 저장됩니다
user_q = st.chat_input("질문을 입력하세요 (예: 외상 환자 지혈은 어떻게?)")

# 사용자가 질문을 입력했을 때만 이 블록이 실행됩니다
if user_q:
    # ========================================
    # 1단계: 사용자 메시지를 대화 기록에 저장하고 화면에 표시
    # ========================================
    # 대화 기록에 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": user_q})
    # 화면에 사용자 메시지 표시 (오른쪽에 표시됨)
    with st.chat_message("user"):
        st.markdown(user_q)

    # ========================================
    # 2단계: 문서에서 관련 내용 검색
    # ========================================
    # retriever.search(): 질문과 관련된 문서 조각들을 검색
    # top_k=top_k: 사이드바에서 설정한 개수만큼 가져옴 (기본 5개)
    # 반환값: RetrievedChunk 객체들의 리스트 (유사도 점수 높은 순으로 정렬됨)
    retrieved = retriever.search(user_q, top_k=top_k)
    
    # 검색 결과가 있으면 가장 높은 유사도 점수를 가져오고, 없으면 0.0
    # retrieved[0]은 유사도가 가장 높은 첫 번째 결과입니다
    best = retrieved[0].score if retrieved else 0.0

    # ========================================
    # 3단계: 검색 결과의 유사도가 낮으면 "추가 질문" 모드로 전환
    # ========================================
    # best < min_sim: 가장 높은 유사도 점수가 설정한 임계값보다 낮은 경우
    # 이 경우 관련 정보가 부족하다고 판단하고, 사용자에게 더 구체적으로 질문하도록 요청합니다
    if best < min_sim:
        # 추가 질문을 요청하는 메시지 생성
        assistant_text = (
            "문서 근거를 충분히 찾기 어렵습니다. 상황을 조금만 더 구체화해 주세요.\n\n"
            "추가 질문 1개: 환자 상태/상황(예: 외상/심정지/호흡곤란/분만/약물 등) 중 어느 범주인가요?"
        )
        # 출처 정보를 마크다운 형식으로 변환 (표시 설정이 켜져있고 검색 결과가 있으면)
        sources_md = citations_markdown(retrieved) if (show_sources and retrieved) else None

        # 대화 기록에 추가하고 화면에 표시
        st.session_state.messages.append({"role": "assistant", "content": assistant_text, "sources_md": sources_md})
        with st.chat_message("assistant"):
            st.markdown(assistant_text)
            if show_sources and sources_md:
                with st.expander("근거 보기"):
                    st.markdown(sources_md)
        st.stop()  # 여기서 종료 (AI 답변 생성하지 않음)

    # ========================================
    # 4단계: 프롬프트 구성 (검색한 문서 조각들을 포함한 질문 만들기)
    # ========================================
    # build_prompt(): 사용자 질문과 검색한 문서 조각들을 합쳐서 AI에게 보낼 프롬프트를 만듭니다
    # 프롬프트에는 다음이 포함됩니다:
    # - 시스템 규칙 (예: "문서 내용만 사용해라", "출처를 명시해라")
    # - 사용자 질문
    # - 검색한 문서 조각들 (CONTEXT)
    prompt = build_prompt(user_q, retrieved)

    # ========================================
    # 5단계: 로컬 LLM (Ollama)을 호출하여 답변 생성
    # ========================================
    # 답변 생성 중 로딩 표시를 위해 assistant 메시지 컨테이너를 먼저 생성
    with st.chat_message("assistant"):
        # 로딩 메시지를 표시할 플레이스홀더 생성
        message_placeholder = st.empty()
        message_placeholder.markdown("답변을 준비하는 중...")
        
        try:
            # ollama.generate(): Ollama 서버에 프롬프트를 보내고 답변을 받아옵니다
            # prompt: 위에서 만든 프롬프트
            # temperature: 사이드바에서 설정한 값 (기본 0.2, 낮을수록 일관적인 답변)
            # st.spinner()로 로딩 스피너 표시
            with st.spinner("AI가 문서를 분석하고 답변을 생성하고 있습니다."):
                resp = ollama.generate(prompt=prompt, temperature=temperature)
            # 응답 텍스트에서 앞뒤 공백을 제거하고, 빈 응답이면 "(빈 응답)" 표시
            assistant_text = resp.text.strip() or "(빈 응답)"
        except Exception as e:
            # Ollama 연결 실패 시 (서버가 꺼져있거나, 모델이 없거나 등)
            # 사용자에게 친절한 에러 메시지 표시
            assistant_text = f"Ollama 호출에 실패했습니다: {e}\n\n- Ollama가 실행 중인지 확인하세요.\n- 모델이 pull 되어 있는지 확인하세요."
        
        # 로딩 메시지를 실제 답변으로 교체
        message_placeholder.markdown(assistant_text)
        
        # 출처 정보를 마크다운 형식으로 변환 (표시 설정이 켜져있으면)
        sources_md = citations_markdown(retrieved) if show_sources else None
        
        if show_sources and sources_md:
            # 출처 정보가 있으면 접을 수 있는 섹션으로 표시
            with st.expander("근거 보기"):
                st.markdown(sources_md)

    # ========================================
    # 6단계: 답변을 대화 기록에 저장
    # ========================================
    # 대화 기록에 AI 답변 추가
    st.session_state.messages.append({"role": "assistant", "content": assistant_text, "sources_md": sources_md})
