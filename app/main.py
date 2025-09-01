from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.tts import router as tts_router
from app.api.advanced_zonos import router as advanced_zonos_router

app = FastAPI(
    title="Hing TTS API - Advanced ZONOS Edition",
    description="""
    고급 Text-to-Speech REST API 서비스
    
    ## 지원 모델
    - **VibeVoice**: Microsoft의 다국어 TTS 모델
    - **Advanced ZONOS**: 한국어 최적화 고품질 TTS 모델
    
    ## 주요 기능
    - 🎭 **감정 제어**: 7가지 프리셋 + 커스텀 벡터
    - 🎤 **Voice Cloning**: 고품질 화자 복제
    - 🇰🇷 **한국어 최적화**: 완전한 한국어 지원
    - ⚡ **배치 처리**: 다중 텍스트 동시 처리
    - 🔄 **지능형 캐싱**: 성능 최적화
    """,
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 기본 TTS 라우터 (하위 호환성)
app.include_router(tts_router, prefix="/api/v1")

# Advanced ZONOS TTS 라우터 (새로운 고급 API)
app.include_router(advanced_zonos_router)

@app.get("/")
async def root():
    return {"message": "Hing TTS API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}