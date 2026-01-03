
"""rag/prompting.py

RAG 답변 품질을 결정하는 것은 '프롬프트(규격)'입니다.
여기서는 다음을 강제합니다:

- 제공된 CONTEXT(근거 조각) 밖의 내용을 단정하지 말 것
- 반드시 '근거' 섹션에 페이지/섹션을 명시할 것
- 근거가 부족하면 '추가 질문 1개'를 우선 제시할 것

주의:
- 의료/응급 처치 관련이므로 안전 문구를 포함합니다.
"""

from __future__ import annotations

from typing import List
from dataclasses import dataclass

from .retriever import RetrievedChunk


def build_prompt(user_question: str, retrieved: List[RetrievedChunk]) -> str:
    # 근거 조각을 보기 좋게 포맷
    context_blocks = []
    for i, ch in enumerate(retrieved, start=1):
        context_blocks.append(
            f"""[근거 {i}] (score={ch.score:.3f}) source={ch.source} p.{ch.page_start}-{ch.page_end} section={ch.section_path}
{ch.text}
"""
        )
    context = "\n\n".join(context_blocks).strip()

    system_rules = """너는 '119 구급대원 현장응급처치 표준지침(2023 개정)' PDF를 근거로만 답하는 보조 챗봇이다.

규칙(매우 중요):
1) 아래 CONTEXT(근거 조각)에 있는 내용만 사용해서 답해라. 모르면 '문서 근거 부족'이라고 말해라.
2) 답변은 반드시 다음 형식을 지켜라.
   - 요약(1~2줄)
   - 현장 절차(단계형, 필요한 경우만)
   - 주의/금기/예외(있으면)
   - 근거(2~5개): p.xx 형태로 페이지 범위와 근거 번호를 함께 제시
   - 추가 질문(필요할 때만 1개)
3) 근거가 애매하거나 부족하면, 답변을 길게 하지 말고 '추가 질문 1개'를 먼저 해라.
4) 실제 현장 처치는 소속 지침/의료지도/교육을 우선한다는 안전 문구를 마지막에 1줄로 포함해라.
"""

    prompt = f"""{system_rules}

[사용자 질문]
{user_question}

[CONTEXT]
{context}

[답변]
"""
    return prompt
