# 119 구급대원 현장응급처치 표준지침(2023) — 100% 로컬 RAG 챗봇 (Streamlit)

이 프로젝트는 **PDF(표준지침)만 근거로** 질문에 답하고, **근거(페이지/조각)를 함께 제시**하는 RAG 챗봇입니다.
임베딩/벡터DB/검색/생성(LLM)까지 **전부 로컬**에서 동작하도록 구성했습니다.

> 주의: 본 도구는 학습/참고용입니다. 실제 현장 처치는 소속 지침, 의료지도, 법·규정 및 교육을 우선합니다.

---

## 1) 구성요소(전부 로컬)

- **PDF 파서**: PyMuPDF (페이지 단위 추출/페이지 메타데이터 확보)
- **임베딩(오픈소스)**: Sentence-Transformers + `intfloat/multilingual-e5-base`
- **벡터DB**: FAISS (CPU)
- **로컬 LLM**: Ollama (HTTP API로 호출)
- **UI**: Streamlit

---

## 2) 빠른 시작

### A. Python 환경 준비
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### B. 로컬 LLM 준비 (Ollama)
1) Ollama 설치 (공식 설치 파일 사용)
2) 모델 다운로드(예시):
```bash
ollama pull llama3.1:8b-instruct
```
- 더 가벼운 모델이 필요하면: `qwen2.5:7b-instruct`, `gemma2:9b-instruct` 등을 시도해도 됩니다.

### C. 인덱스(벡터DB) 생성
```bash
python -m ingest.build_index --pdf data/guide_2023.pdf --out artifacts
```
성공하면 다음 파일이 생성됩니다:
- `artifacts/faiss.index`
- `artifacts/meta.json`
- `artifacts/chunks.jsonl`

### D. 실행
```bash
streamlit run app.py
```

---

## 3) 동작 개요

1) 질문 입력
2) (Retriever) 임베딩 → FAISS에서 Top-k 근거 조각 검색
3) (Guardrail) 근거가 약하면 “추가 질문 1개”로 전환
4) (Generator) 근거 조각만 컨텍스트로 로컬 LLM(Ollama) 호출
5) 답변 + 근거(페이지/섹션/스니펫) 출력

---

## 4) 설정(중요)

Streamlit 좌측 패널에서:
- Top-k
- 유사도 임계값
- Ollama 모델명
- 근거 표시 여부
를 조정할 수 있습니다.

---

## 5) 폴더 구조

- `data/` 원본 PDF
- `ingest/` PDF → 전처리/청킹/인덱싱
- `rag/` retriever, ollama client, 프롬프트/포맷터
- `artifacts/` faiss.index, meta.json, chunks.jsonl
- `app.py` Streamlit 앱

---

## 6) 오프라인 완전 차단에 대해
임베딩 모델(`multilingual-e5-base`)은 **최초 1회 HuggingFace에서 다운로드**가 필요합니다.
다운로드 후에는 캐시를 사용하므로 오프라인에서도 동작합니다.
완전 오프라인을 원하면, 인터넷 되는 환경에서 모델을 미리 받아서 캐시 디렉토리를 옮기는 방식으로 처리합니다.
