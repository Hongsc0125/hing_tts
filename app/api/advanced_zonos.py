"""
Advanced ZONOS TTS API - 한국어 최적화 완전 구현
==================================================

완전히 새로운 Advanced ZONOS TTS API 엔드포인트들을 제공합니다.
기존 API와 완전히 분리된 고급 기능들을 포함합니다.
"""

import os
import json
from typing import List, Optional, Union
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from app.services.advanced_zonos_tts_service import AdvancedZonosTTSService

router = APIRouter(prefix="/api/tts/advanced", tags=["Advanced ZONOS TTS"])

# 전역 Advanced ZONOS 서비스 인스턴스
advanced_zonos_service = None

def get_advanced_zonos_service():
    """Advanced ZONOS 서비스 인스턴스 가져오기 (Lazy Loading)"""
    global advanced_zonos_service
    if advanced_zonos_service is None:
        advanced_zonos_service = AdvancedZonosTTSService(model_type="transformer")
    return advanced_zonos_service


# Pydantic 모델들
class AdvancedTTSRequest(BaseModel):
    """고급 TTS 생성 요청"""
    text: str = Field(..., description="생성할 텍스트", min_length=1, max_length=1000)
    emotion: Union[str, List[float]] = Field("neutral", description="감정 ('neutral', 'happy', 'sad', etc.) 또는 8차원 커스텀 벡터")
    speaker_name: Optional[str] = Field(None, description="화자명 (None이면 기본 화자)")
    cfg_scale: float = Field(2.5, description="생성 품질 제어 (1.0-5.0)", ge=1.0, le=5.0)
    pitch_std: Optional[float] = Field(None, description="피치 변화폭 (0-400)", ge=0, le=400)
    speaking_rate: Optional[float] = Field(None, description="발화 속도 (0-40)", ge=0, le=40)
    fmax: Optional[float] = Field(None, description="최대 주파수 (0-24000)", ge=0, le=24000)


class BatchTTSRequest(BaseModel):
    """배치 TTS 생성 요청"""
    texts: List[str] = Field(..., description="생성할 텍스트 목록", min_items=1, max_items=10)
    emotions: Optional[List[Union[str, List[float]]]] = Field(None, description="각 텍스트별 감정")
    speaker_names: Optional[List[str]] = Field(None, description="각 텍스트별 화자명")
    cfg_scale: float = Field(2.5, description="생성 품질 제어 (1.0-5.0)", ge=1.0, le=5.0)


class SpeakerEmbeddingRequest(BaseModel):
    """Speaker embedding 생성 요청"""
    audio_file_path: str = Field(..., description="오디오 파일 경로")


def cleanup_audio_file(file_path: str):
    """생성된 오디오 파일 정리"""
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(f"⚠️ 파일 정리 실패: {e}")


@router.post("/generate", 
             summary="고품질 한국어 음성 생성",
             description="Advanced ZONOS TTS를 사용한 고품질 한국어 음성 생성")
async def generate_advanced_speech(
    request: AdvancedTTSRequest, 
    background_tasks: BackgroundTasks
):
    """
    고품질 한국어 음성 생성
    
    - **한국어 직접 지원**: ko 언어 코드 사용
    - **감정 제어**: 7가지 프리셋 또는 8차원 커스텀 벡터
    - **Voice Cloning**: 고품질 Speaker embedding
    - **44.1kHz 출력**: 네이티브 고음질
    """
    try:
        service = get_advanced_zonos_service()
        
        audio_path, metadata = service.generate_speech(
            text=request.text,
            emotion=request.emotion,
            speaker_name=request.speaker_name,
            cfg_scale=request.cfg_scale,
            pitch_std=request.pitch_std,
            speaking_rate=request.speaking_rate,
            fmax=request.fmax
        )
        
        # 파일 정리 태스크 등록
        background_tasks.add_task(cleanup_audio_file, audio_path)
        
        filename = f"advanced_zonos_{hash(request.text) % 10000}.wav"
        
        return FileResponse(
            path=audio_path,
            media_type="audio/wav",
            filename=filename,
            headers={
                "X-Audio-Duration": str(metadata.duration),
                "X-Sample-Rate": str(metadata.sample_rate),
                "X-File-Size": str(metadata.file_size),
                "X-Generation-Time": str(metadata.generation_time),
                "X-Model-Config": json.dumps(metadata.model_config)
            }
        )
        
    except Exception as e:
        print(f"❌ Advanced ZONOS TTS 생성 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Advanced ZONOS TTS 생성 실패: {str(e)}")


@router.post("/batch",
             summary="배치 음성 생성", 
             description="여러 텍스트를 한 번에 처리하여 음성 생성")
async def generate_batch_speech(
    request: BatchTTSRequest,
    background_tasks: BackgroundTasks
):
    """
    배치 음성 생성
    
    - **다중 처리**: 최대 10개 텍스트 동시 처리
    - **개별 설정**: 텍스트별 다른 감정/화자 적용 가능
    - **효율적**: 배치 처리로 성능 최적화
    """
    try:
        service = get_advanced_zonos_service()
        
        results = service.generate_batch(
            texts=request.texts,
            emotions=request.emotions,
            speaker_names=request.speaker_names,
            cfg_scale=request.cfg_scale
        )
        
        # 성공한 결과들만 수집
        successful_results = []
        for i, (audio_path, metadata) in enumerate(results):
            if audio_path and metadata:
                # 파일 정리 태스크 등록
                background_tasks.add_task(cleanup_audio_file, audio_path)
                
                successful_results.append({
                    "index": i,
                    "text": request.texts[i],
                    "audio_url": f"/download/temp/{os.path.basename(audio_path)}",
                    "duration": metadata.duration,
                    "file_size": metadata.file_size,
                    "generation_time": metadata.generation_time
                })
        
        return {
            "total_requests": len(request.texts),
            "successful_count": len(successful_results),
            "results": successful_results
        }
        
    except Exception as e:
        print(f"❌ Advanced ZONOS 배치 생성 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"배치 생성 실패: {str(e)}")


@router.get("/voices",
           summary="한국어 음성 목록",
           description="사용 가능한 한국어 화자 음성 목록 반환")
async def list_advanced_voices():
    """
    한국어 음성 목록 반환
    
    - **한국어 특화**: 한국어 화자만 포함
    - **Voice Cloning**: 각 화자별 고유한 음성 특성
    """
    try:
        service = get_advanced_zonos_service()
        voices = service.list_korean_voices()
        
        return {
            "model": "Advanced ZONOS TTS",
            "language": "Korean (ko)",
            "total_voices": len(voices),
            "voices": voices
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"음성 목록 조회 실패: {str(e)}")


@router.get("/emotions",
           summary="감정 프리셋 목록",
           description="사용 가능한 감정 프리셋과 8차원 벡터 반환")
async def list_emotion_presets():
    """
    감정 프리셋 목록 반환
    
    - **7가지 프리셋**: neutral, happy, sad, angry, surprised, calm, expressive
    - **8차원 벡터**: Happiness, Sadness, Disgust, Fear, Surprise, Anger, Other, Neutral
    - **커스텀 지원**: 직접 8차원 벡터 입력 가능
    """
    try:
        service = get_advanced_zonos_service()
        presets = service.list_emotion_presets()
        
        return {
            "emotion_dimensions": [
                "Happiness", "Sadness", "Disgust", "Fear", 
                "Surprise", "Anger", "Other", "Neutral"
            ],
            "presets": presets,
            "usage_examples": {
                "string_emotion": "happy",
                "custom_vector": [0.6, 0.05, 0.05, 0.05, 0.1, 0.05, 0.1, 0.05],
                "description": "각 차원의 값은 0.0-1.0 범위이며, 합계는 자동으로 1.0으로 정규화됩니다"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"감정 프리셋 조회 실패: {str(e)}")


@router.get("/info",
           summary="모델 정보",
           description="Advanced ZONOS TTS 모델의 상세 정보 반환")
async def get_advanced_model_info():
    """
    모델 정보 반환
    
    - **모델 상태**: 로드 여부, 디바이스 정보
    - **성능 지표**: 캐시 크기, 음성 수
    - **최적화 설정**: 한국어 최적화 파라미터
    """
    try:
        service = get_advanced_zonos_service()
        info = service.get_model_info()
        
        return {
            "model_name": "Advanced ZONOS TTS",
            "version": "v0.1-transformer",
            "language_support": "Korean (ko) + 126 others",
            "sample_rate": "44.1kHz",
            "quality": "High-fidelity",
            "features": [
                "Voice Cloning",
                "Emotion Control", 
                "Korean Optimization",
                "Batch Processing",
                "Smart Caching"
            ],
            **info
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모델 정보 조회 실패: {str(e)}")


@router.post("/estimate-time",
            summary="생성 시간 추정",
            description="텍스트 길이 기반으로 음성 생성 소요 시간 추정")
async def estimate_generation_time(text: str):
    """
    음성 생성 시간 추정
    
    - **정확한 예측**: 한국어 텍스트 분석 기반
    - **GPU 최적화**: Real-time factor ~2x 고려
    - **사전 계획**: 대용량 작업 계획 수립 도움
    """
    try:
        service = get_advanced_zonos_service()
        estimation = service.estimate_generation_time(text)
        
        return {
            "input_text": text[:100] + "..." if len(text) > 100 else text,
            **estimation
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"시간 추정 실패: {str(e)}")


@router.delete("/cache",
              summary="캐시 초기화", 
              description="Speaker embedding 캐시를 완전히 초기화")
async def clear_cache():
    """
    캐시 초기화
    
    - **메모리 최적화**: Speaker embedding 캐시 정리
    - **성능 관리**: 메모리 사용량 감소
    """
    try:
        service = get_advanced_zonos_service()
        service.clear_cache()
        
        return {
            "status": "success",
            "message": "Speaker embedding 캐시가 초기화되었습니다"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"캐시 초기화 실패: {str(e)}")


@router.post("/speaker-embedding",
            summary="Speaker embedding 생성",
            description="오디오 파일에서 speaker embedding 생성 및 캐싱")
async def create_speaker_embedding(request: SpeakerEmbeddingRequest):
    """
    Speaker embedding 생성
    
    - **Voice Cloning**: 고품질 화자 특성 추출
    - **자동 캐싱**: 재사용시 빠른 로딩
    - **전처리**: 16kHz 리샘플링, 모노 변환 자동 적용
    """
    try:
        service = get_advanced_zonos_service()
        
        if not os.path.exists(request.audio_file_path):
            raise HTTPException(status_code=404, detail="오디오 파일을 찾을 수 없습니다")
        
        embedding = service._create_speaker_embedding(request.audio_file_path)
        
        if embedding is None:
            raise HTTPException(status_code=500, detail="Speaker embedding 생성 실패")
        
        return {
            "status": "success",
            "file_path": request.audio_file_path,
            "embedding_shape": list(embedding.shape),
            "cached": True,
            "message": "Speaker embedding이 생성되고 캐시되었습니다"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Speaker embedding 생성 실패: {str(e)}")