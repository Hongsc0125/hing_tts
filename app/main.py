from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.tts import router as tts_router

app = FastAPI(
    title="Hing TTS API",
    description="Text-to-Speech REST API Service",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TTS 라우터 포함
app.include_router(tts_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Hing TTS API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}