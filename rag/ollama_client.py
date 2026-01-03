
"""rag/ollama_client.py

Ollama 로컬 서버(기본: http://localhost:11434)로 프롬프트를 보내 응답을 받습니다.

- Ollama가 실행 중이어야 합니다.
- 모델은 `ollama pull <model>`로 미리 받아둬야 합니다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Any, Optional

import requests

from config import SETTINGS


@dataclass
class OllamaResponse:
    text: str
    raw: Dict[str, Any]


class OllamaClient:
    def __init__(self, base_url: str = SETTINGS.ollama_base_url, model: str = SETTINGS.ollama_model, timeout_s: int = 120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def generate(self, prompt: str, system: Optional[str] = None, temperature: float = 0.2) -> OllamaResponse:
        # Ollama /api/generate 엔드포인트 사용
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        # system 메시지를 강하게 주고 싶다면 prompt 상단에 포함하는 방식도 가능하지만,
        # Ollama 모델별 호환 차이가 있어 여기서는 prompt 내부에서 처리합니다.
        r = requests.post(url, json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()
        return OllamaResponse(text=data.get("response", ""), raw=data)
