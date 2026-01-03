"""
rag/ollama_client.py

이 파일은 Ollama 로컬 서버와 통신하여 AI 모델의 답변을 받아오는 클라이언트를 담고 있습니다.

Ollama는 로컬 컴퓨터에서 AI 모델을 실행할 수 있게 해주는 도구입니다.
이 파일의 OllamaClient 클래스는 Ollama 서버에 HTTP 요청을 보내고 응답을 받아옵니다.

사전 요구사항:
- Ollama가 실행 중이어야 합니다 (백그라운드에서 돌아가고 있어야 함)
- 사용할 모델은 `ollama pull <model>` 명령어로 미리 다운로드되어 있어야 합니다
  예: ollama pull llama3.1:8b-instruct

Ollama 기본 주소: http://localhost:11434 (기본 포트)
"""

from __future__ import annotations  # 미래 버전 Python 호환성

import json  # JSON 데이터 처리용 (현재는 사용 안 하지만 향후 확장용)
from dataclasses import dataclass  # 데이터클래스 정의용
from typing import Dict, Any, Optional  # 타입 힌팅용

import requests  # HTTP 요청을 보내기 위한 라이브러리

from config import SETTINGS  # 프로젝트 설정 (Ollama 주소, 모델 이름 등)


@dataclass
class OllamaResponse:
    """
    Ollama 서버로부터 받은 응답을 저장하는 데이터클래스
    
    Attributes:
        text: AI가 생성한 답변 텍스트 (사용자에게 보여줄 최종 답변)
        raw: Ollama 서버가 반환한 전체 응답 데이터 (디버깅이나 추가 정보 필요 시 사용)
    """
    text: str  # AI가 생성한 답변 텍스트
    raw: Dict[str, Any]  # Ollama 서버의 원본 응답 (딕셔너리 형태)


class OllamaClient:
    """
    Ollama 서버와 통신하는 클라이언트 클래스
    
    이 클래스는 Ollama 서버에 HTTP 요청을 보내서 AI 모델의 답변을 받아옵니다.
    """
    
    def __init__(self, base_url: str = SETTINGS.ollama_base_url, model: str = SETTINGS.ollama_model, timeout_s: int = 120):
        """
        OllamaClient 객체를 초기화합니다.
        
        Args:
            base_url: Ollama 서버의 기본 주소 (예: "http://localhost:11434")
                     기본값은 config.py의 SETTINGS에서 가져옵니다
            model: 사용할 Ollama 모델 이름 (예: "llama3.1:8b-instruct")
                  기본값은 config.py의 SETTINGS에서 가져옵니다
                  이 모델은 미리 다운로드되어 있어야 합니다
            timeout_s: 요청 타임아웃 시간 (초 단위)
                      AI 답변 생성에 시간이 오래 걸릴 수 있으므로 기본값은 120초 (2분)
        """
        # .rstrip("/"): URL 끝의 슬래시(/)를 제거 (일관성 유지)
        # 예: "http://localhost:11434/" → "http://localhost:11434"
        self.base_url = base_url.rstrip("/")
        self.model = model  # 사용할 모델 이름 저장
        self.timeout_s = timeout_s  # 타임아웃 시간 저장

    def generate(self, prompt: str, system: Optional[str] = None, temperature: float = 0.2) -> OllamaResponse:
        """
        Ollama 서버에 프롬프트를 보내고 AI 답변을 받아옵니다.
        
        이 함수는 Ollama의 /api/generate 엔드포인트를 사용합니다.
        프롬프트를 Ollama 서버에 보내면, AI 모델이 답변을 생성하여 반환합니다.
        
        Args:
            prompt: AI에게 보낼 프롬프트 (질문 + 문서 내용이 포함된 전체 텍스트)
            system: 시스템 메시지 (현재는 사용하지 않음, None)
                   참고: 일부 Ollama 모델은 system 메시지를 지원하지만,
                         모델별 호환 차이가 있어 여기서는 prompt 내부에 규칙을 포함하는 방식을 사용
            temperature: 답변의 창의성/랜덤성을 조절하는 값 (0.0 ~ 1.0)
                        0.0에 가까울수록 일관적이고 예측 가능한 답변
                        1.0에 가까울수록 다양하고 창의적인 답변
                        의료/응급처치 같은 정확성이 중요한 분야에서는 낮은 값(0.2)을 권장
        
        Returns:
            OllamaResponse: AI가 생성한 답변을 담은 객체
                          - text: 답변 텍스트
                          - raw: 원본 응답 데이터
        
        Raises:
            requests.exceptions.RequestException: Ollama 서버 연결 실패 시
            requests.exceptions.Timeout: 타임아웃 발생 시
            requests.exceptions.HTTPError: HTTP 에러 발생 시 (예: 모델이 없을 때)
        """
        # Ollama의 /api/generate 엔드포인트 URL 구성
        url = f"{self.base_url}/api/generate"
        
        # Ollama 서버에 보낼 요청 데이터 (JSON 형태)
        payload = {
            "model": self.model,  # 사용할 모델 이름
            "prompt": prompt,  # AI에게 보낼 프롬프트
            "stream": False,  # False: 전체 답변을 한 번에 받음, True: 실시간으로 스트리밍
            "options": {
                "temperature": temperature,  # 답변의 창의성 조절
                "num_predict": 512,  # 최대 생성 토큰 수 제한 (기본값보다 낮게 설정하여 속도 향상)
            },
        }
        
        # 참고: system 메시지를 별도로 보내는 방법도 있지만,
        # Ollama 모델별 호환 차이가 있어 여기서는 prompt 내부에 규칙을 포함하는 방식을 사용합니다.
        # prompt 상단에 시스템 규칙을 명시하는 것이 더 안정적입니다.
        
        # HTTP POST 요청 보내기
        # requests.post(): POST 요청을 보내는 함수
        # url: 요청을 보낼 주소
        # json=payload: 요청 본문을 JSON 형식으로 보냄 (자동으로 Content-Type 헤더도 설정됨)
        # timeout=self.timeout_s: 타임아웃 시간 설정
        r = requests.post(url, json=payload, timeout=self.timeout_s)
        
        # r.raise_for_status(): HTTP 응답 상태 코드가 에러(4xx, 5xx)인 경우 예외를 발생시킴
        # 예: 404 (모델 없음), 500 (서버 에러) 등
        r.raise_for_status()
        
        # r.json(): 응답 본문을 JSON 형식으로 파싱하여 딕셔너리로 변환
        data = r.json()
        
        # OllamaResponse 객체 생성하여 반환
        # data.get("response", ""): 응답 데이터에서 "response" 키의 값을 가져옴, 없으면 빈 문자열
        # data: 전체 응답 데이터를 raw에 저장 (디버깅 등에 유용)
        return OllamaResponse(text=data.get("response", ""), raw=data)
