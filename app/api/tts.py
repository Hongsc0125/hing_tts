import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from app.models.tts_request import TTSRequest
from app.services.tts_service import tts_service

router = APIRouter()


@router.post("/generate")
async def generate_speech(request: TTSRequest):
    """
    VibeVoice 로컬 모델을 사용하여 한국어 TTS 생성
    
    Args:
        request: 텍스트, 스피커 이름 목록, CFG 스케일을 포함한 TTSRequest
        
    Returns:
        오디오 파일 응답 (24kHz WAV, 한국어 음성 샘플 기반)
    """
    try:
        print(f"🎙️ 한국어 TTS 요청: {request.text[:50]}...")
        
        # VibeVoice로 음성 파일 생성 (한국어 음성 샘플 자동 사용)
        audio_path = tts_service.generate_speech(
            text=request.text,
            speaker_names=request.speaker_names,
            cfg_scale=request.cfg_scale
        )
        
        # BackgroundTask를 사용한 파일 정리
        task = BackgroundTask(os.unlink, audio_path)
        
        # 자동 정리 기능과 함께 오디오 파일 반환
        return FileResponse(
            path=audio_path,
            media_type="audio/wav",
            filename=f"korean_tts_{hash(request.text) % 10000}.wav",
            background=task
        )
        
    except Exception as e:
        print(f"❌ 한국어 TTS 생성 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"한국어 TTS 생성 실패: {str(e)}")


@router.get("/voices")
async def list_voices():
    """
    사용 가능한 한국어 음성 목록 반환
    """
    try:
        voices = tts_service.list_korean_voices()
        return {"voices": voices}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"음성 목록 조회 실패: {str(e)}")