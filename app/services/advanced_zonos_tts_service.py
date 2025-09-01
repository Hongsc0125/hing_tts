"""
Advanced ZONOS TTS Service - 한국어 최적화 완전 구현
=======================================================

ZONOS TTS 모델의 완전한 분석 결과를 바탕으로 한국어에 최적화된 
고급 TTS 서비스를 구현합니다.

주요 기능:
- 한국어 직접 지원 (ko 언어 코드)
- 정교한 감정 제어 (7가지 프리셋 + 커스텀)
- 고품질 Voice Cloning
- 지능형 캐싱 시스템
- 배치 처리 지원
"""

import os
import sys
import tempfile
import torch
import torchaudio
import numpy as np
from pathlib import Path
import soundfile as sf
from typing import List, Optional, Dict, Union, Tuple
import traceback
import time
import hashlib
import json
from dataclasses import dataclass

# ZONOS 모듈 경로 추가
zonos_path = "/home/hsc0125/Hing_tts/models/Zonos"
if zonos_path not in sys.path:
    sys.path.insert(0, zonos_path)

try:
    from zonos.model import Zonos
    from zonos.conditioning import make_cond_dict
    from zonos.utils import DEFAULT_DEVICE
    ZONOS_AVAILABLE = True
    print("✅ Advanced ZONOS 라이브러리 로드 성공")
except ImportError as e:
    print(f"❌ Advanced ZONOS 라이브러리 로드 실패: {e}")
    ZONOS_AVAILABLE = False


@dataclass
class AudioMetadata:
    """생성된 오디오의 메타데이터"""
    duration: float
    sample_rate: int
    file_size: int
    generation_time: float
    model_config: dict


class SpeakerEmbeddingCache:
    """Speaker embedding 캐싱 시스템"""
    
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size
        self.access_times = {}
    
    def _get_key(self, audio_path: str) -> str:
        """오디오 파일 경로 기반 캐시 키 생성"""
        return hashlib.md5(audio_path.encode()).hexdigest()
    
    def get(self, audio_path: str) -> Optional[torch.Tensor]:
        """캐시에서 speaker embedding 조회"""
        key = self._get_key(audio_path)
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
    
    def put(self, audio_path: str, embedding: torch.Tensor):
        """Speaker embedding을 캐시에 저장"""
        key = self._get_key(audio_path)
        
        # 캐시 크기 초과시 가장 오래된 항목 제거
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[key] = embedding.clone()
        self.access_times[key] = time.time()
        print(f"🔄 Speaker embedding 캐시됨: {os.path.basename(audio_path)}")
    
    def clear(self):
        """캐시 완전 초기화"""
        self.cache.clear()
        self.access_times.clear()
        print("🗑️ Speaker embedding 캐시 초기화됨")


class AdvancedZonosTTSService:
    """Advanced ZONOS TTS Service - 한국어 최적화"""
    
    # 한국어 최적화 감정 프리셋
    EMOTION_PRESETS = {
        "neutral": [0.15, 0.05, 0.05, 0.05, 0.05, 0.05, 0.2, 0.4],    # 중성 - 뉴스/내레이션
        "happy": [0.6, 0.05, 0.05, 0.05, 0.1, 0.05, 0.1, 0.05],       # 기쁨 - 광고/활발한 음성
        "sad": [0.05, 0.5, 0.05, 0.05, 0.05, 0.05, 0.25, 0.05],       # 슬픔 - 감성적 표현
        "angry": [0.05, 0.05, 0.05, 0.05, 0.05, 0.6, 0.2, 0.05],      # 분노 - 강한 표현
        "surprised": [0.05, 0.05, 0.05, 0.05, 0.6, 0.05, 0.2, 0.05],  # 놀라움 - 리액션
        "calm": [0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.55],       # 차분함 - 명상/진중
        "expressive": [0.25, 0.1, 0.05, 0.1, 0.15, 0.1, 0.2, 0.05]    # 표현력 - 연기/드라마
    }
    
    # 한국어 최적화 기본 설정
    KOREAN_OPTIMAL_CONFIG = {
        "language": "ko",              # 한국어 직접 지원
        "fmax": 22050.0,              # Voice cloning 권장값  
        "pitch_std": 30.0,            # 한국어 자연스러운 억양
        "speaking_rate": 13.0,        # 적당한 발화 속도
        "cfg_scale": 2.5,             # 안정적 생성 품질
        "max_new_tokens_per_char": 8, # 텍스트 길이 대비 토큰 수
    }
    
    def __init__(self, model_type: str = "transformer"):
        """
        Advanced ZONOS TTS 서비스 초기화
        
        Args:
            model_type: "transformer" 또는 "hybrid" (hybrid는 더 높은 품질, 더 많은 리소스 필요)
        """
        self.model_type = model_type
        self.model_path = "/home/hsc0125/Hing_tts/models/Zonos"
        self.audio_samples_path = "/home/hsc0125/Hing_tts/models/audio_data"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.korean_voices = {}
        self.embedding_cache = SpeakerEmbeddingCache(max_size=50)
        
        print(f"🎙️ Advanced ZONOS TTS 서비스 초기화 중... 디바이스: {self.device}, 모델: {model_type}")
        self._initialize_model()
        self._load_korean_voices()
    
    def _initialize_model(self):
        """ZONOS TTS 모델 초기화"""
        if not ZONOS_AVAILABLE:
            print("⚠️ ZONOS 라이브러리가 로드되지 않았습니다.")
            self.model = None
            return
            
        try:
            print(f"📁 ZONOS {self.model_type} 모델 로딩...")
            
            # 모델 타입에 따른 repo ID 선택
            repo_id = f"Zyphra/Zonos-v0.1-{self.model_type}"
            
            # ZONOS 모델 로드 (Hugging Face에서 자동 다운로드)
            self.model = Zonos.from_pretrained(repo_id, device=self.device)
            self.model.requires_grad_(False).eval()
            
            print(f"✅ Advanced ZONOS {self.model_type} 모델 초기화 완료")
            print(f"🎯 한국어 최적화 설정 적용됨")
            
        except Exception as e:
            print(f"❌ Advanced ZONOS 모델 로드 실패: {e}")
            print("⚠️ 더미 모드로 대체합니다.")
            traceback.print_exc()
            self.model = None
    
    def _load_korean_voices(self):
        """한국어 음성 샘플들을 로드"""
        try:
            audio_data_path = Path(self.audio_samples_path)
            print(f"🔍 오디오 경로 확인: {audio_data_path}")
            
            if not audio_data_path.exists():
                print(f"⚠️ 오디오 샘플 폴더를 찾을 수 없습니다: {self.audio_samples_path}")
                audio_data_path.mkdir(parents=True, exist_ok=True)
                print("📁 오디오 폴더 생성됨")
                return
            
            # 고품질 오디오 파일 우선 순위
            audio_extensions = ['*.wav', '*.flac', '*.mp3', '*.ogg']
            audio_files = []
            
            for ext in audio_extensions:
                found_files = list(audio_data_path.glob(ext))
                audio_files.extend(found_files)
            
            print(f"🔍 발견된 오디오 파일: {len(audio_files)}개")
            for f in audio_files:
                print(f"  - {f.name}")
            
            if not audio_files:
                print("⚠️ 오디오 파일이 없습니다. 기본 음성으로 진행합니다.")
                return
            
            # 한국어 화자명 매핑
            korean_speaker_names = [
                "한국여성1", "한국여성2", "한국여성3", "한국여성4",
                "한국남성1", "한국남성2", "한국남성3", "한국남성4"
            ]
            
            for i, audio_file in enumerate(audio_files[:len(korean_speaker_names)]):
                speaker_name = korean_speaker_names[i]
                self.korean_voices[speaker_name] = str(audio_file)
                print(f"🎤 Advanced ZONOS 한국어 음성 등록: {speaker_name} -> {audio_file.name}")
            
            # 첫 번째 파일을 기본값으로 설정
            if audio_files:
                self.korean_voices["default"] = str(audio_files[0])
                print(f"✅ Advanced ZONOS 기본 음성 설정: {audio_files[0].name}")
                
        except Exception as e:
            print(f"❌ Advanced ZONOS 한국어 음성 샘플 로드 실패: {e}")
            traceback.print_exc()
    
    def _create_speaker_embedding(self, audio_path: str) -> Optional[torch.Tensor]:
        """Speaker embedding 생성 (캐싱 지원)"""
        if not self.model:
            return None
            
        # 캐시에서 확인
        cached_embedding = self.embedding_cache.get(audio_path)
        if cached_embedding is not None:
            print(f"🔄 캐시된 Speaker embedding 사용: {os.path.basename(audio_path)}")
            return cached_embedding
        
        try:
            print(f"🎤 Speaker embedding 생성 중: {os.path.basename(audio_path)}")
            
            # 오디오 로드 및 전처리
            wav, sr = torchaudio.load(audio_path)
            
            # 모노 변환
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            
            # 16kHz로 리샘플링 (ZONOS 권장)
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                wav = resampler(wav)
            
            # 길이 제한 (최대 30초)
            max_samples = 16000 * 30
            if wav.shape[1] > max_samples:
                wav = wav[:, :max_samples]
                print(f"  📏 오디오 길이 조정: {wav.shape[1] / 16000:.1f}초")
            
            # Speaker embedding 생성
            embedding = self.model.make_speaker_embedding(wav, 16000)
            
            # 캐시에 저장
            self.embedding_cache.put(audio_path, embedding)
            
            print(f"✅ Speaker embedding 생성 완료: {embedding.shape}")
            return embedding
            
        except Exception as e:
            print(f"❌ Speaker embedding 생성 실패: {e}")
            return None
    
    def list_korean_voices(self) -> List[str]:
        """사용 가능한 한국어 음성 목록 반환"""
        return list(self.korean_voices.keys())
    
    def list_emotion_presets(self) -> Dict[str, List[float]]:
        """사용 가능한 감정 프리셋 목록 반환"""
        return self.EMOTION_PRESETS.copy()
    
    def get_model_info(self) -> Dict:
        """모델 정보 반환"""
        return {
            "model_type": self.model_type,
            "device": self.device,
            "korean_voices_count": len(self.korean_voices),
            "emotion_presets_count": len(self.EMOTION_PRESETS),
            "cache_size": len(self.embedding_cache.cache),
            "optimal_config": self.KOREAN_OPTIMAL_CONFIG,
            "is_loaded": self.model is not None
        }
    
    def estimate_generation_time(self, text: str) -> Dict:
        """음성 생성 예상 시간 계산"""
        text_length = len(text)
        char_per_second = 15  # 한국어 기준 평균
        estimated_audio_duration = text_length / char_per_second
        
        # GPU 성능에 따른 생성 시간 (Real-time factor ~2x)
        generation_time = estimated_audio_duration / 2.0
        
        return {
            "text_length": text_length,
            "estimated_audio_duration": estimated_audio_duration,
            "estimated_generation_time": generation_time,
            "real_time_factor": 2.0
        }
    
    def generate_speech(
        self, 
        text: str, 
        emotion: Union[str, List[float]] = "neutral",
        speaker_name: Optional[str] = None,
        cfg_scale: float = 2.5,
        pitch_std: Optional[float] = None,
        speaking_rate: Optional[float] = None,
        fmax: Optional[float] = None
    ) -> Tuple[str, AudioMetadata]:
        """
        고품질 한국어 음성 생성
        
        Args:
            text: 생성할 텍스트
            emotion: 감정 ("neutral", "happy", "sad", etc.) 또는 커스텀 8차원 벡터
            speaker_name: 화자명 (None이면 기본 화자)
            cfg_scale: 생성 품질 제어 (1.0-5.0, 높을수록 고품질)
            pitch_std: 피치 변화폭 (기본값: 30.0)
            speaking_rate: 발화 속도 (기본값: 13.0)
            fmax: 최대 주파수 (기본값: 22050.0)
            
        Returns:
            (audio_file_path, metadata)
        """
        if not self.model:
            raise Exception("Advanced ZONOS TTS 모델이 로드되지 않음")
        
        start_time = time.time()
        
        # 임시 출력 파일 생성
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            output_path = tmp_file.name
        
        try:
            print(f"🎬 Advanced ZONOS TTS 음성 생성 시작: {text[:50]}...")
            
            # 감정 설정 처리
            if isinstance(emotion, str):
                if emotion not in self.EMOTION_PRESETS:
                    print(f"⚠️ 알 수 없는 감정 '{emotion}', 'neutral'로 대체")
                    emotion = "neutral"
                emotion_vector = self.EMOTION_PRESETS[emotion]
                print(f"🎭 감정 설정: {emotion}")
            else:
                if len(emotion) != 8:
                    raise ValueError("커스텀 감정 벡터는 8차원이어야 합니다")
                emotion_vector = emotion
                print(f"🎭 커스텀 감정 벡터 적용: {emotion_vector}")
            
            # 화자 설정
            speaker_embedding = None
            if self.korean_voices:
                voice_path = None
                
                if speaker_name and speaker_name in self.korean_voices:
                    voice_path = self.korean_voices[speaker_name]
                    print(f"🎤 지정된 화자: {speaker_name}")
                elif "default" in self.korean_voices:
                    voice_path = self.korean_voices["default"]
                    print(f"🎤 기본 화자 사용")
                else:
                    voice_path = list(self.korean_voices.values())[0]
                    print(f"🎤 첫 번째 화자 사용")
                
                if voice_path:
                    speaker_embedding = self._create_speaker_embedding(voice_path)
            
            # 설정값 최적화 적용
            config = self.KOREAN_OPTIMAL_CONFIG.copy()
            if pitch_std is not None:
                config["pitch_std"] = pitch_std
            if speaking_rate is not None:
                config["speaking_rate"] = speaking_rate
            if fmax is not None:
                config["fmax"] = fmax
            
            # Conditioning dictionary 생성
            cond_dict = make_cond_dict(
                text=text,
                language=config["language"],
                speaker=speaker_embedding,
                emotion=emotion_vector,
                fmax=config["fmax"],
                pitch_std=config["pitch_std"],
                speaking_rate=config["speaking_rate"],
                device=self.device
            )
            
            print(f"🔧 설정값 - 피치: {config['pitch_std']}, 속도: {config['speaking_rate']}, CFG: {cfg_scale}")
            
            # Conditioning 준비
            conditioning = self.model.prepare_conditioning(cond_dict)
            
            # 음성 생성
            print("🎵 Advanced ZONOS로 고품질 음성 생성 중...")
            max_tokens = int(len(text) * config["max_new_tokens_per_char"])
            
            with torch.inference_mode():
                codes = self.model.generate(
                    conditioning,
                    cfg_scale=cfg_scale,
                    max_new_tokens=max_tokens,
                    progress_bar=True,
                    disable_torch_compile=True  # 안정성 확보
                )
            
            # 오디오 디코딩
            print("🔄 44.1kHz 고품질 오디오 디코딩 중...")
            wavs = self.model.autoencoder.decode(codes).cpu()
            
            # WAV 파일로 저장 (ZONOS는 44.1kHz 네이티브)
            sample_rate = self.model.autoencoder.sampling_rate
            torchaudio.save(output_path, wavs[0], sample_rate)
            
            # 메타데이터 생성
            generation_time = time.time() - start_time
            file_size = os.path.getsize(output_path)
            duration = wavs[0].shape[1] / sample_rate
            
            metadata = AudioMetadata(
                duration=duration,
                sample_rate=sample_rate,
                file_size=file_size,
                generation_time=generation_time,
                model_config={
                    "model_type": self.model_type,
                    "emotion": emotion if isinstance(emotion, str) else "custom",
                    "cfg_scale": cfg_scale,
                    "pitch_std": config["pitch_std"],
                    "speaking_rate": config["speaking_rate"],
                    "fmax": config["fmax"]
                }
            )
            
            print(f"💾 Advanced ZONOS 음성 파일 저장 완료: {output_path}")
            print(f"📊 음성 길이: {duration:.2f}초, 파일 크기: {file_size/1024:.1f}KB, 생성 시간: {generation_time:.2f}초")
            print(f"⚡ Real-time factor: {duration/generation_time:.1f}x")
            
            return output_path, metadata
            
        except Exception as e:
            print(f"❌ Advanced ZONOS TTS 음성 생성 실패: {e}")
            traceback.print_exc()
            if os.path.exists(output_path):
                os.unlink(output_path)
            raise e
    
    def generate_batch(
        self,
        texts: List[str],
        emotions: Optional[List[Union[str, List[float]]]] = None,
        speaker_names: Optional[List[str]] = None,
        cfg_scale: float = 2.5
    ) -> List[Tuple[str, AudioMetadata]]:
        """
        배치 음성 생성 - 여러 텍스트를 한 번에 처리
        
        Args:
            texts: 생성할 텍스트 목록
            emotions: 각 텍스트별 감정 (None이면 모두 neutral)
            speaker_names: 각 텍스트별 화자명 (None이면 모두 기본 화자)
            cfg_scale: 생성 품질 제어
            
        Returns:
            [(audio_file_path, metadata), ...] 목록
        """
        print(f"🔄 배치 처리 시작: {len(texts)}개 텍스트")
        
        results = []
        total_start_time = time.time()
        
        # 기본값 설정
        if emotions is None:
            emotions = ["neutral"] * len(texts)
        if speaker_names is None:
            speaker_names = [None] * len(texts)
        
        # 길이 맞추기
        emotions = emotions[:len(texts)] + ["neutral"] * (len(texts) - len(emotions))
        speaker_names = speaker_names[:len(texts)] + [None] * (len(texts) - len(speaker_names))
        
        for i, (text, emotion, speaker_name) in enumerate(zip(texts, emotions, speaker_names)):
            try:
                print(f"🔄 배치 {i+1}/{len(texts)} 처리 중...")
                audio_path, metadata = self.generate_speech(
                    text=text,
                    emotion=emotion,
                    speaker_name=speaker_name,
                    cfg_scale=cfg_scale
                )
                results.append((audio_path, metadata))
                print(f"✅ 배치 {i+1}/{len(texts)} 완료")
                
            except Exception as e:
                print(f"❌ 배치 {i+1}/{len(texts)} 실패: {e}")
                results.append((None, None))
        
        total_time = time.time() - total_start_time
        successful_count = sum(1 for result in results if result[0] is not None)
        
        print(f"🎉 배치 처리 완료: {successful_count}/{len(texts)} 성공, 총 시간: {total_time:.2f}초")
        
        return results
    
    def clear_cache(self):
        """Speaker embedding 캐시 초기화"""
        self.embedding_cache.clear()