
"""app.py

Streamlit 기반 로컬 RAG 챗봇.
- 임베딩/검색: 로컬
- 생성(LLM): Ollama 로컬

실행 전:
1) python -m ingest.build_index --pdf data/guide_2023.pdf --out artifacts
2) ollama pull llama3.1:8b-instruct
3) streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import streamlit as st

from config import SETTINGS
from rag.retriever import FaissRetriever, RetrievedChunk
from rag.ollama_client import OllamaClient
from rag.prompting import build_prompt
from rag.formatting import citations_markdown


ARTIFACTS_DIR = Path("artifacts")


# -----------------------------
# Streamlit 캐시(무거운 리소스)
# -----------------------------

@st.cache_resource
def load_retriever() -> FaissRetriever:
    return FaissRetriever(ARTIFACTS_DIR, embedding_model_name=SETTINGS.embedding_model_name)


@st.cache_resource
def load_ollama(model_name: str) -> OllamaClient:
    return OllamaClient(model=model_name)


# -----------------------------
# UI
# -----------------------------

st.set_page_config(page_title="119 현장응급처치 RAG", layout="wide")
st.title("119 구급대원 현장응급처치 RAG 챗봇 (로컬)")

with st.sidebar:
    st.header("설정")
    top_k = st.slider("Top-k(근거 조각 개수)", min_value=3, max_value=10, value=SETTINGS.top_k, step=1)
    min_sim = st.slider("최소 유사도(이하면 되묻기)", min_value=0.10, max_value=0.60, value=SETTINGS.min_similarity, step=0.01)
    show_sources = st.checkbox("근거(페이지/조각) 표시", value=True)
    ollama_model = st.text_input("Ollama 모델명", value=SETTINGS.ollama_model)
    temperature = st.slider("LLM temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

    st.divider()
    st.caption("인덱스 파일이 없으면 먼저 build_index를 실행하세요.")


# 인덱스 존재 체크
if not (ARTIFACTS_DIR / "faiss.index").exists():
    st.error("artifacts/faiss.index 가 없습니다. 먼저 인덱스를 생성하세요.")
    st.code("python -m ingest.build_index --pdf data/guide_2023.pdf --out artifacts", language="bash")
    st.stop()


retriever = load_retriever()
ollama = load_ollama(ollama_model)

if "messages" not in st.session_state:
    st.session_state.messages = []  # [{role, content, sources?}]

# 기존 대화 표시
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if show_sources and m.get("sources_md"):
            with st.expander("근거 보기"):
                st.markdown(m["sources_md"])

user_q = st.chat_input("질문을 입력하세요 (예: 외상 환자 지혈은 어떻게?)")

if user_q:
    # 1) 사용자 메시지 저장/표시
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    # 2) 검색
    retrieved = retriever.search(user_q, top_k=top_k)
    best = retrieved[0].score if retrieved else 0.0

    # 3) 근거가 약하면: 추가 질문 모드 (양방향)
    if best < min_sim:
        assistant_text = (
            "문서 근거를 충분히 찾기 어렵습니다. 상황을 조금만 더 구체화해 주세요.\n\n"
            "추가 질문 1개: 환자 상태/상황(예: 외상/심정지/호흡곤란/분만/약물 등) 중 어느 범주인가요?"
        )
        sources_md = citations_markdown(retrieved) if (show_sources and retrieved) else None

        st.session_state.messages.append({"role": "assistant", "content": assistant_text, "sources_md": sources_md})
        with st.chat_message("assistant"):
            st.markdown(assistant_text)
            if show_sources and sources_md:
                with st.expander("근거 보기"):
                    st.markdown(sources_md)
        st.stop()

    # 4) 프롬프트 구성(근거 조각만 사용)
    prompt = build_prompt(user_q, retrieved)

    # 5) 로컬 LLM 호출
    try:
        resp = ollama.generate(prompt=prompt, temperature=temperature)
        assistant_text = resp.text.strip() or "(빈 응답)"
    except Exception as e:
        assistant_text = f"Ollama 호출에 실패했습니다: {e}\n\n- Ollama가 실행 중인지 확인하세요.\n- 모델이 pull 되어 있는지 확인하세요."

    sources_md = citations_markdown(retrieved) if show_sources else None

    # 6) 저장/표시
    st.session_state.messages.append({"role": "assistant", "content": assistant_text, "sources_md": sources_md})
    with st.chat_message("assistant"):
        st.markdown(assistant_text)
        if show_sources and sources_md:
            with st.expander("근거 보기"):
                st.markdown(sources_md)
