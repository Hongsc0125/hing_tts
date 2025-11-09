from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.tts import router as tts_router

app = FastAPI(
    title="ChatterBox TTS API",
    description="""
    ResembleAI ChatterBox 기반 고품질 Text-to-Speech API

    ## ChatterBox 모델 특징
    - 🌍 **23개 언어 지원**: 한국어, 영어, 중국어, 일본어, 프랑스어 등
    - 🎭 **감정 제어**: exaggeration 파라미터로 감정 강도 조절
    - 🎚️ **품질 제어**: CFG 스케일로 생성 품질 조절
    - 🎤 **제로샷 음성 복제**: 샘플 없이도 음성 복제 가능
    - 🔊 **24kHz 고품질**: 프로페셔널급 오디오 출력

    ## API 엔드포인트
    - `POST /generate`: 텍스트를 고품질 음성으로 변환
    - `GET /languages`: 지원 언어 목록 조회
    - `GET /info`: 모델 정보 및 파라미터 가이드

    ## 사용 예시
    ```json
    {
      "text": "안녕하세요, ChatterBox TTS입니다!",
      "language_id": "ko",
      "exaggeration": 0.7,
      "cfg": 0.5,
      "temperature": 1.0
    }
    ```
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

# ChatterBox TTS 라우터
app.include_router(tts_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Hing TTS API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}