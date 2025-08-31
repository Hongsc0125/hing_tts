import os
import sys
import tempfile
import torch
import random
import numpy as np
from pathlib import Path
import soundfile as sf
from typing import List, Optional
import traceback

# VibeVoice 모듈 경로 추가
vibevoice_path = "/home/hsc0125/Hing_tts/models"
if vibevoice_path not in sys.path:
    sys.path.insert(0, vibevoice_path)

try:
    from VibeVoice.vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
    from VibeVoice.vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
    from VibeVoice.vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
    VIBEVOICE_AVAILABLE = True
    print("✅ VibeVoice 로컬 라이브러리 로드 성공")
except ImportError as e:
    print(f"❌ VibeVoice 로컬 라이브러리 로드 실패: {e}")
    VIBEVOICE_AVAILABLE = False


class TTSService:
    def __init__(self):
        self.model_path = "/home/hsc0125/Hing_tts/models/VibeVoice-1.5B"
        self.audio_samples_path = "/home/hsc0125/Hing_tts/models/audio_data"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self.korean_voices = {}
        print(f"🎙️ VibeVoice TTS 서비스 초기화 중... 디바이스: {self.device}")
        self._initialize_model()
        self._load_korean_voices()
    
    def _initialize_model(self):
        """VibeVoice 모델 초기화"""
        if not VIBEVOICE_AVAILABLE:
            raise Exception("VibeVoice 라이브러리가 로드되지 않았습니다.")
        
        try:
            print(f"📁 VibeVoice 모델 로딩...")
            
            # 프로세서 로드
            self.processor = VibeVoiceProcessor.from_pretrained(
                "microsoft/VibeVoice-1.5B",
                trust_remote_code=True
            )
            print("✅ VibeVoice 프로세서 로드 성공")
            
            # 모델 로드
            self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                "microsoft/VibeVoice-1.5B",
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map='cuda' if self.device == "cuda" else None,
                trust_remote_code=True,
                attn_implementation="eager"  # 안정성을 위해 eager 사용
            )
            
            self.model.eval()
            print("✅ VibeVoice 모델 로드 성공")
            
            # DDPM 설정 (공식 데모 기본값)
            if hasattr(self.model, 'set_ddmp_inference_steps'):
                self.model.set_ddmp_inference_steps(num_steps=5)
                print("✅ DDPM 추론 스텝 설정: 5")
            
            print("🎉 VibeVoice 모델 초기화 완료")
            
        except Exception as e:
            print(f"❌ VibeVoice 모델 로드 실패: {e}")
            traceback.print_exc()
            raise e
    
    def _load_korean_voices(self):
        """한국어 음성 샘플들을 로드"""
        try:
            audio_data_path = Path(self.audio_samples_path)
            print(f"🔍 오디오 경로 확인: {audio_data_path}")
            
            if not audio_data_path.exists():
                print(f"⚠️ 오디오 샘플 폴더를 찾을 수 없습니다: {self.audio_samples_path}")
                # 폴더가 없으면 생성
                audio_data_path.mkdir(parents=True, exist_ok=True)
                print("📁 오디오 폴더 생성됨")
                return
            
            # 모든 오디오 파일 확인 (.wav, .mp3 등)
            audio_files = []
            for ext in ['*.wav', '*.mp3', '*.flac', '*.ogg']:
                audio_files.extend(list(audio_data_path.glob(ext)))
            
            print(f"🔍 발견된 오디오 파일: {len(audio_files)}개")
            for f in audio_files:
                print(f"  - {f.name}")
            
            if not audio_files:
                print("⚠️ 오디오 파일이 없습니다. 기본 음성으로 진행합니다.")
                return
            
            korean_speaker_names = ["한국여성1", "한국여성2", "한국남성1", "한국남성2", "한국여성3"]
            
            for i, audio_file in enumerate(audio_files[:len(korean_speaker_names)]):
                speaker_name = korean_speaker_names[i]
                self.korean_voices[speaker_name] = str(audio_file)
                print(f"🎤 한국어 음성 등록: {speaker_name} -> {audio_file.name}")
            
            # 첫 번째 파일을 기본값으로 설정
            if audio_files:
                self.korean_voices["default"] = str(audio_files[0])
                print(f"✅ 기본 음성 설정: {audio_files[0].name}")
                
        except Exception as e:
            print(f"❌ 한국어 음성 샘플 로드 실패: {e}")
            traceback.print_exc()
    
    def list_korean_voices(self) -> List[str]:
        """사용 가능한 한국어 음성 목록 반환"""
        return list(self.korean_voices.keys())
    
    def generate_speech(self, text: str, speaker_names: List[str] = None, cfg_scale: float = 3.0) -> str:
        """
        VibeVoice로 텍스트를 음성으로 변환
        """
        if not self.model or not self.processor:
            raise Exception("VibeVoice 모델이 초기화되지 않음")
        
        # 임시 출력 파일 생성
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            output_path = tmp_file.name
        
        try:
            print(f"🎬 음성 생성 시작: {text[:50]}...")
            
            # 간단한 스크립트 포맷팅
            formatted_script = f"Speaker 0: {text}"
            
            # 한국어 음성 샘플 준비
            voice_samples = []
            if self.korean_voices:
                if "default" in self.korean_voices:
                    voice_path = self.korean_voices["default"]
                else:
                    voice_path = list(self.korean_voices.values())[0]
                voice_samples = [voice_path]
                print(f"🎤 한국어 음성 샘플 사용: {voice_path}")
            else:
                print("⚠️ 한국어 음성 샘플이 없습니다. 기본 VibeVoice 음성을 사용합니다.")
            
            # VibeVoice 처리 - 음성 샘플 포함
            print(f"🔄 텍스트 처리: {formatted_script}")
            if voice_samples:
                print(f"🎤 음성 샘플 경로: {voice_samples}")
                inputs = self.processor(
                    text=[formatted_script],
                    voice_samples=[voice_samples],
                    padding=True,
                    return_tensors="pt",
                    return_attention_mask=True
                )
            else:
                # 음성 샘플이 없으면 기본 처리
                print("⚠️ 음성 샘플 없이 처리")
                inputs = self.processor(
                    text=[formatted_script],
                    padding=True,
                    return_tensors="pt",
                    return_attention_mask=True
                )
            
            print(f"디버그 - inputs 키들: {list(inputs.keys())}")
            for key, value in inputs.items():
                print(f"  {key}: type={type(value)}, is_tensor={torch.is_tensor(value) if value is not None else False}")
            
            # GPU로 이동
            if self.device == "cuda":
                for key, value in inputs.items():
                    if value is not None and torch.is_tensor(value):
                        try:
                            inputs[key] = value.to(self.device)
                            print(f"  ✅ {key} GPU로 이동 완료")
                        except Exception as e:
                            print(f"  ❌ {key} GPU 이동 실패: {e}")
                    else:
                        print(f"  ⏭️ {key} 스킵 (None 또는 비텐서)")
            
            # 음성 생성
            print(f"🎵 음성 생성 중... (CFG: {cfg_scale})")
            
            # None 값들을 필터링하여 전달
            filtered_inputs = {k: v for k, v in inputs.items() if v is not None}
            print(f"모델에 전달할 inputs: {list(filtered_inputs.keys())}")
            
            try:
                with torch.no_grad():
                    outputs = self.model.generate(
                        **filtered_inputs,
                        cfg_scale=cfg_scale,
                        tokenizer=self.processor.tokenizer
                    )
            except Exception as gen_error:
                print(f"🔍 모델 생성 중 상세 오류:")
                traceback.print_exc()
                raise gen_error
            
            # 오디오 추출 및 저장
            if hasattr(outputs, 'speech_outputs') and outputs.speech_outputs:
                audio_data = outputs.speech_outputs[0]
                
                if torch.is_tensor(audio_data):
                    if audio_data.dtype == torch.bfloat16:
                        audio_data = audio_data.float()
                    audio_array = audio_data.cpu().detach().numpy()
                else:
                    audio_array = np.array(audio_data, dtype=np.float32)
                
                # 차원 정리
                while audio_array.ndim > 1 and audio_array.shape[0] == 1:
                    audio_array = audio_array.squeeze(0)
                if audio_array.ndim > 1:
                    audio_array = audio_array.flatten()
                
                # 정규화
                max_val = max(abs(audio_array.min()), abs(audio_array.max()))
                if max_val > 1.0:
                    audio_array = audio_array / max_val * 0.95
                
                # 저장
                sf.write(output_path, audio_array, 24000)
                print(f"💾 음성 파일 저장 완료: {output_path}")
                return output_path
            else:
                raise Exception("음성 출력을 찾을 수 없음")
            
        except Exception as e:
            print(f"❌ 음성 생성 실패: {e}")
            if os.path.exists(output_path):
                os.unlink(output_path)
            raise e


# 전역 TTS 서비스 인스턴스
tts_service = TTSService()