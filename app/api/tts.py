import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from app.models.tts_request import TTSRequest, ModelType
from app.services.tts_service import TTSServiceFactory

router = APIRouter()


@router.post("/generate")
async def generate_speech(request: TTSRequest):
    """
    선택된 모델을 사용하여 TTS 생성 (VibeVoice 또는 ZONOS)
    
    Args:
        request: 텍스트, 모델 타입, 스피커 이름 목록, CFG 스케일을 포함한 TTSRequest
        
    Returns:
        오디오 파일 응답 (24kHz WAV)
    """
    try:
        model_name = request.model_type.value.upper()
        print(f"🎙️ {model_name} TTS 요청: {request.text[:50]}...")
        
        # 선택된 모델의 TTS 서비스 가져오기
        tts_service = TTSServiceFactory.get_service(request.model_type)
        
        # 선택된 모델로 음성 파일 생성
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
            filename=f"{model_name.lower()}_tts_{hash(request.text) % 10000}.wav",
            background=task
        )
        
    except Exception as e:
        print(f"❌ {model_name} TTS 생성 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"{model_name} TTS 생성 실패: {str(e)}")


@router.get("/voices")
async def list_voices(model_type: ModelType = ModelType.VIBEVOICE):
    """
    선택된 모델의 사용 가능한 한국어 음성 목록 반환
    """
    try:
        tts_service = TTSServiceFactory.get_service(model_type)
        voices = tts_service.list_korean_voices()
        return {"model_type": model_type.value, "voices": voices}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"음성 목록 조회 실패: {str(e)}")


@router.get("/models")
async def list_models():
    """
    사용 가능한 TTS 모델 목록 반환
    """
    return {
        "models": [
            {"id": ModelType.VIBEVOICE.value, "name": "VibeVoice-1.5B", "description": "Microsoft VibeVoice 모델 (영어/중국어 네이티브, 한국어 음성샘플)"},
            {"id": ModelType.ZONOS.value, "name": "ZONOS TTS", "description": "ZONOS TTS 모델"}
        ],
        "advanced_api": {
            "endpoint": "/api/tts/advanced/*",
            "description": "고급 ZONOS TTS API - 한국어 최적화, 감정 제어, voice cloning 지원"
        }
    }