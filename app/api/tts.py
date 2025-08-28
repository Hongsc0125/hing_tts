import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from app.models.tts_request import TTSRequest
from app.services.tts_service import tts_service

router = APIRouter()


@router.post("/input")
async def generate_speech(request: TTSRequest):
    """
    한국어 텍스트 입력으로부터 음성 생성
    
    Args:
        request: 텍스트와 선택적 스피커 음성 파일을 포함한 TTSRequest
        
    Returns:
        오디오 파일 응답
    """
    try:
        # 한국어 음성 파일 생성
        audio_path = tts_service.generate_speech(
            text=request.text,
            speaker_wav=request.speaker_wav
        )
        
        # 자동 정리 기능과 함께 오디오 파일 반환
        return FileResponse(
            path=audio_path,
            media_type="audio/wav",
            filename="generated_speech.wav",
            background=lambda: os.unlink(audio_path) if os.path.exists(audio_path) else None
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"음성 생성 실패: {str(e)}")