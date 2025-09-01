import os
import sys
import tempfile
import torch
import torchaudio
import numpy as np
from pathlib import Path
import soundfile as sf
from typing import List, Optional
import traceback

# Zonos 모듈 경로 추가
zonos_path = "/home/hsc0125/Hing_tts/models/Zonos"
if zonos_path not in sys.path:
    sys.path.insert(0, zonos_path)

try:
    from zonos.model import Zonos
    from zonos.conditioning import make_cond_dict
    from zonos.utils import DEFAULT_DEVICE
    ZONOS_AVAILABLE = True
    print("✅ Zonos 라이브러리 로드 성공")
except ImportError as e:
    print(f"❌ Zonos 라이브러리 로드 실패: {e}")
    ZONOS_AVAILABLE = False


class ZonosTTSService:
    def __init__(self):
        self.model_path = "/home/hsc0125/Hing_tts/models/Zonos"
        self.audio_samples_path = "/home/hsc0125/Hing_tts/models/audio_data"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.korean_voices = {}
        print(f"🎙️ ZONOS TTS 서비스 초기화 중... 디바이스: {self.device}")
        self._initialize_model()
        self._load_korean_voices()
    
    def _initialize_model(self):
        """ZONOS TTS 모델 초기화"""
        if not ZONOS_AVAILABLE:
            print("⚠️ Zonos 라이브러리가 로드되지 않았습니다. 더미 모드로 실행합니다.")
            self.model = "dummy_zonos_model"
            return
            
        try:
            print(f"📁 ZONOS TTS 모델 로딩...")
            
            # Zonos 모델 로드 (Hugging Face에서 자동 다운로드)
            # transformer 모델 사용 (hybrid는 더 많은 리소스 필요)
            self.model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=self.device)
            self.model.requires_grad_(False).eval()
            
            print("✅ ZONOS TTS 모델 초기화 완료")
            
        except Exception as e:
            print(f"❌ ZONOS TTS 모델 로드 실패: {e}")
            print("⚠️ 더미 모드로 대체합니다.")
            traceback.print_exc()
            self.model = "dummy_zonos_model"
    
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
            
            # 모든 오디오 파일 확인 (.wav, .mp3 등)
            audio_files = []
            for ext in ['*.wav', '*.mp3', '*.flac', '*.ogg']:
                audio_files.extend(list(audio_data_path.glob(ext)))
            
            print(f"🔍 발견된 오디오 파일: {len(audio_files)}개")
            
            if not audio_files:
                print("⚠️ 오디오 파일이 없습니다. 기본 음성으로 진행합니다.")
                return
            
            korean_speaker_names = ["한국여성1", "한국여성2", "한국남성1", "한국남성2", "한국여성3"]
            
            for i, audio_file in enumerate(audio_files[:len(korean_speaker_names)]):
                speaker_name = korean_speaker_names[i]
                self.korean_voices[speaker_name] = str(audio_file)
                print(f"🎤 ZONOS 한국어 음성 등록: {speaker_name} -> {audio_file.name}")
            
            # 첫 번째 파일을 기본값으로 설정
            if audio_files:
                self.korean_voices["default"] = str(audio_files[0])
                print(f"✅ ZONOS 기본 음성 설정: {audio_files[0].name}")
                
        except Exception as e:
            print(f"❌ ZONOS 한국어 음성 샘플 로드 실패: {e}")
            traceback.print_exc()
    
    def list_korean_voices(self) -> List[str]:
        """사용 가능한 한국어 음성 목록 반환"""
        return list(self.korean_voices.keys())
    
    def generate_speech(self, text: str, speaker_names: List[str] = None, cfg_scale: float = 2.0) -> str:
        """
        ZONOS TTS로 텍스트를 음성으로 변환
        """
        if not self.model:
            raise Exception("ZONOS TTS 모델이 초기화되지 않음")
        
        # 임시 출력 파일 생성
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            output_path = tmp_file.name
        
        try:
            print(f"🎬 ZONOS TTS 음성 생성 시작: {text[:50]}...")
            
            if self.model == "dummy_zonos_model":
                # 더미 구현: 사인파 오디오 생성
                sample_rate = 44100  # Zonos는 44kHz 출력
                duration = min(len(text) * 0.1, 10.0)
                t = np.linspace(0, duration, int(sample_rate * duration))
                frequency = 440
                audio_data = 0.3 * np.sin(2 * np.pi * frequency * t)
                sf.write(output_path, audio_data, sample_rate)
                print(f"💾 ZONOS TTS 더미 음성 파일 저장 완료: {output_path}")
                return output_path
            
            # 실제 ZONOS TTS 구현
            # 한국어 음성 샘플 로드 (speaker embedding 생성용)
            speaker_embedding = None
            if self.korean_voices:
                try:
                    # 첫 번째 한국어 음성 샘플을 speaker embedding으로 사용
                    if "default" in self.korean_voices:
                        voice_path = self.korean_voices["default"]
                    else:
                        voice_path = list(self.korean_voices.values())[0]
                    
                    print(f"🎤 한국어 음성 샘플 로드: {voice_path}")
                    wav, sr = torchaudio.load(voice_path)
                    speaker_embedding = self.model.make_speaker_embedding(wav, sr)
                    print("✅ Speaker embedding 생성 완료")
                except Exception as e:
                    print(f"⚠️ Speaker embedding 생성 실패: {e}")
            
            # Conditioning dictionary 생성
            # Zonos는 "ko" 대신 "en-us"를 사용 (한국어 직접 지원 안됨, 영어로 대체)
            cond_dict = make_cond_dict(
                text=text,
                language="en-us",  # 영어로 설정 (한국어 미지원)
                speaker=speaker_embedding,
                # 자연스러운 감정
                emotion=[0.2, 0.05, 0.05, 0.05, 0.05, 0.05, 0.3, 0.25],  
                fmax=22050.0,  # Voice cloning에 권장되는 값
                pitch_std=25.0,  # 적당한 pitch variation
                speaking_rate=12.0,  # 적당한 속도
                device=self.device
            )
            
            # Conditioning 준비
            conditioning = self.model.prepare_conditioning(cond_dict)
            
            # 음성 생성
            print("🎵 ZONOS로 음성 생성 중...")
            with torch.inference_mode():
                codes = self.model.generate(
                    conditioning,
                    cfg_scale=cfg_scale,
                    max_new_tokens=int(len(text) * 8),  # 텍스트 길이에 비례
                    progress_bar=True,
                    disable_torch_compile=True  # C++ 컴파일러 문제 방지
                )
            
            # 오디오 디코딩
            print("🔄 오디오 디코딩 중...")
            wavs = self.model.autoencoder.decode(codes).cpu()
            
            # WAV 파일로 저장 (Zonos는 44kHz로 출력)
            torchaudio.save(output_path, wavs[0], self.model.autoencoder.sampling_rate)
            
            print(f"💾 ZONOS TTS 음성 파일 저장 완료: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ ZONOS TTS 음성 생성 실패: {e}")
            traceback.print_exc()
            if os.path.exists(output_path):
                os.unlink(output_path)
            raise e