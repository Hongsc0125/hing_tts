from pydantic import BaseModel
from typing import Optional
from enum import Enum


class LanguageId(str, Enum):
    """ChatterBox 지원 언어 (23개 언어)"""
    ENGLISH = "en"
    KOREAN = "ko"
    CHINESE = "zh"
    JAPANESE = "ja"
    FRENCH = "fr"
    GERMAN = "de"
    SPANISH = "es"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    ARABIC = "ar"
    HINDI = "hi"
    # 기타 언어들도 추가 가능
    AUTO = "auto"  # 자동 감지


class TTSRequest(BaseModel):
    """ChatterBox TTS 요청 모델"""
    text: str
    language_id: Optional[LanguageId] = LanguageId.AUTO
    exaggeration: Optional[float] = 0.5  # 감정 강도 (0.0-1.0)
    cfg: Optional[float] = 0.5  # 생성 품질 제어 (0.0-1.0)
    temperature: Optional[float] = 1.0  # 다양성 제어 (0.1-2.0)

    class Config:
        # Pydantic 경고 해결
        protected_namespaces = ()