from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    # 임베딩 모델: 다국어 검색에 안정적인 E5 계열(오픈소스)
    embedding_model_name: str = "intfloat/multilingual-e5-base"

    # Ollama 로컬 서버 기본 주소
    ollama_base_url: str = "http://localhost:11434"

    # 기본 LLM (Ollama에서 pull 해 둔 모델명과 동일해야 함)
    ollama_model: str = "llama3.1:8b-instruct"

    # 기본 검색 Top-k
    top_k: int = 5

    # 코사인 유사도 임계값(정규화된 임베딩의 inner product)
    # 값이 낮으면 헛근거로 답할 위험이 커짐 → 임계값 미만이면 "추가 질문" 모드
    min_similarity: float = 0.30

SETTINGS = Settings()
