import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from app.models.tts_request import TTSRequest, LanguageId
from app.services.chatterbox_service import TTSServiceFactory

router = APIRouter()


@router.post("/generate")
async def generate_speech(request: TTSRequest):
    """
    ChatterBox TTS를 사용하여 음성 생성

    Args:
        request: ChatterBox TTS 요청 (텍스트, 언어, 감정강도, CFG, 온도)

    Returns:
        고품질 오디오 파일 (24kHz WAV)
    """
    try:
        print(f"🎙️ ChatterBox TTS 요청: {request.text[:50]}...")
        print(f"📋 언어: {request.language_id}, 감정: {request.exaggeration}, CFG: {request.cfg}")

        # ChatterBox TTS 서비스 가져오기
        tts_service = TTSServiceFactory.get_service()

        # ChatterBox 모델로 음성 파일 생성
        audio_path = tts_service.generate_speech(
            text=request.text,
            language_id=request.language_id.value,
            exaggeration=request.exaggeration,
            cfg=request.cfg,
            temperature=request.temperature
        )

        # BackgroundTask를 사용한 파일 정리
        task = BackgroundTask(os.unlink, audio_path)

        # 자동 정리 기능과 함께 오디오 파일 반환
        return FileResponse(
            path=audio_path,
            media_type="audio/wav",
            filename=f"chatterbox_tts_{hash(request.text) % 10000}.wav",
            background=task
        )

    except Exception as e:
        print(f"❌ ChatterBox TTS 생성 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ChatterBox TTS 생성 실패: {str(e)}")


@router.get("/languages")
async def list_languages():
    """
    ChatterBox TTS가 지원하는 언어 목록 반환 (23개 언어)
    """
    try:
        tts_service = TTSServiceFactory.get_service()
        languages = tts_service.list_supported_languages()
        return {
            "supported_languages": languages,
            "total_count": len(languages),
            "description": "ChatterBox는 23개 언어를 지원하며, 자동 언어 감지도 가능합니다."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"언어 목록 조회 실패: {str(e)}")


@router.get("/info")
async def get_model_info():
    """
    ChatterBox TTS 모델 정보 및 기능 소개
    """
    return {
        "model": {
            "name": "ChatterBox TTS",
            "provider": "ResembleAI",
            "version": "Latest",
            "description": "23개 언어 지원 고품질 TTS 모델"
        },
        "features": {
            "languages": "23개 언어 지원",
            "emotion_control": "감정 제어 가능 (exaggeration)",
            "quality_control": "CFG 스케일로 품질 조절",
            "voice_cloning": "Zero-shot 음성 복제",
            "sample_rate": "24kHz 고품질 오디오"
        },
        "parameters": {
            "exaggeration": "0.0-1.0 (감정 강도, 기본값: 0.5)",
            "cfg": "0.0-1.0 (품질 제어, 기본값: 0.5)",
            "temperature": "0.1-2.0 (다양성 제어, 기본값: 1.0)",
            "language_id": "auto 또는 지원 언어 코드"
        }
    }