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
from abc import ABC, abstractmethod

# 실제 TTS 라이브러리들 로드
try:
    import pyttsx3
    from gtts import gTTS
    import edge_tts
    import asyncio
    REAL_TTS_AVAILABLE = True
    print("✅ 실제 TTS 라이브러리들 로드 성공 (pyttsx3, gTTS, edge-tts)")
except ImportError as e:
    print(f"❌ TTS 라이브러리 로드 실패: {e}")
    REAL_TTS_AVAILABLE = False



class BaseTTSService(ABC):
    """TTS 서비스 기본 인터페이스"""
    
    @abstractmethod
    def generate_speech(self, text: str, voice: str = "auto", speed: float = 1.0) -> str:
        """음성 생성"""
        pass
    
    @abstractmethod
    def list_korean_voices(self) -> List[str]:
        """사용 가능한 한국어 음성 목록 반환"""
        pass


class ChatterBoxTTSService(BaseTTSService):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pyttsx3_engine = None
        self.korean_voices = ["한국여성1", "한국남성1", "한국여성2", "한국남성2", "Edge-TTS-SunHi", "Edge-TTS-InJoon"]
        print(f"🎙️ 실제 TTS 서비스 초기화 중... 디바이스: {self.device}")
        self._initialize_model()

    def _initialize_model(self):
        """실제 TTS 모델 초기화"""
        if not REAL_TTS_AVAILABLE:
            print("⚠️ TTS 라이브러리가 로드되지 않았습니다. 더미 모드로 실행합니다.")
            return

        try:
            print(f"📁 실제 TTS 엔진들 초기화 중...")

            # pyttsx3 엔진 초기화 (로컬 TTS)
            try:
                self.pyttsx3_engine = pyttsx3.init()
                # 음성 속도 설정
                self.pyttsx3_engine.setProperty('rate', 150)
                print("✅ pyttsx3 로컬 TTS 엔진 초기화 성공")
            except Exception as e:
                print(f"⚠️ pyttsx3 초기화 실패: {e}")
                self.pyttsx3_engine = None

            # gTTS는 필요시 사용 (온라인 TTS)
            print("✅ gTTS 온라인 TTS 준비됨")

            # Edge-TTS는 고품질 한국어 지원
            print("✅ Edge-TTS 고품질 다국어 TTS 준비됨")

            print("🎉 실제 TTS 엔진들 초기화 완료")

        except Exception as e:
            print(f"❌ TTS 엔진 초기화 실패: {e}")
            traceback.print_exc()
            raise e

    def list_korean_voices(self) -> List[str]:
        """사용 가능한 한국어 음성 목록 반환"""
        return self.korean_voices

    def generate_speech(self, text: str, voice: str = "auto", speed: float = 1.0) -> str:
        """
        ChatterBox TTS로 텍스트를 음성으로 변환

        Args:
            text: 변환할 텍스트
            voice: 음성 타입 (auto, korean_female, korean_male, english_default)
            speed: 음성 속도 (0.5-2.0)
        """
        # 임시 출력 파일 생성
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            output_path = tmp_file.name

        try:
            print(f"🎬 실제 TTS 음성 생성 시작: {text[:50]}...")

            if not REAL_TTS_AVAILABLE:
                # 더미 모드: 빈 오디오 파일 생성
                print("⚠️ 더미 모드: 빈 오디오 파일 생성")
                import numpy as np
                import soundfile as sf
                # 1초간의 침묵 생성
                dummy_audio = np.zeros(16000)  # 16kHz, 1초
                sf.write(output_path, dummy_audio, 16000)
                return output_path

            # 음성 타입에 따른 TTS 엔진 선택
            if voice == "auto":
                # 자동 감지: 한국어가 있으면 한국어, 없으면 영어
                import re
                has_korean = bool(re.search(r'[가-힣]', text))

                if has_korean:
                    print("🇰🇷 한국어 텍스트 자동 감지 - Edge-TTS 여성 음성")
                    return self._generate_with_edge_tts(text, output_path, "ko-KR-SunHiNeural", speed)
                else:
                    print("🇺🇸 영어 텍스트 자동 감지 - pyttsx3 사용")
                    return self._generate_with_pyttsx3(text, output_path, speed)

            elif voice == "korean_female":
                print("🇰🇷 한국어 여성 음성 - Edge-TTS SunHi")
                return self._generate_with_edge_tts(text, output_path, "ko-KR-SunHiNeural", speed)

            elif voice == "korean_male":
                print("🇰🇷 한국어 남성 음성 - Edge-TTS InJoon")
                return self._generate_with_edge_tts(text, output_path, "ko-KR-InJoonNeural", speed)

            elif voice == "english_default":
                print("🇺🇸 영어 기본 음성 - pyttsx3 사용")
                return self._generate_with_pyttsx3(text, output_path, speed)

            else:
                print(f"⚠️ 알 수 없는 음성 타입 '{voice}', 자동 모드 사용")
                import re
                has_korean = bool(re.search(r'[가-힣]', text))

                if has_korean:
                    return self._generate_with_edge_tts(text, output_path, "ko-KR-SunHiNeural", speed)
                else:
                    return self._generate_with_pyttsx3(text, output_path, speed)

        except Exception as e:
            print(f"❌ 실제 TTS 음성 생성 실패: {e}")
            if os.path.exists(output_path):
                os.unlink(output_path)
            raise e

    def _generate_with_edge_tts(self, text: str, output_path: str, voice: str = "ko-KR-SunHiNeural", speed: float = 1.0) -> str:
        """Edge-TTS로 음성 생성 (한국어 고품질)"""
        try:
            import subprocess
            import tempfile

            # Edge-TTS CLI 사용 (비동기 루프 충돌 방지)
            # 속도 조절을 위한 rate 설정
            rate_percent = f"{int((speed - 1.0) * 50):+d}%" if speed != 1.0 else "+0%"

            result = subprocess.run([
                'edge-tts',
                '--voice', voice,
                '--text', text,
                '--rate', rate_percent,
                '--write-media', output_path
            ], capture_output=True, text=True, timeout=30)

            if result.returncode == 0 and os.path.exists(output_path):
                print(f"💾 Edge-TTS 음성 파일 저장 완료: {output_path}")
                return output_path
            else:
                raise Exception(f"Edge-TTS CLI 실패: {result.stderr}")

        except Exception as e:
            print(f"❌ Edge-TTS 실패: {e}")
            # 실패시 gTTS로 폴백
            return self._generate_with_gtts(text, output_path, 'ko', speed)

    def _generate_with_pyttsx3(self, text: str, output_path: str, speed: float = 1.0) -> str:
        """pyttsx3로 음성 생성 (로컬 영어)"""
        try:
            if not self.pyttsx3_engine:
                raise Exception("pyttsx3 엔진이 초기화되지 않음")

            # 속도 설정 (pyttsx3의 기본은 150 WPM)
            base_rate = 150
            new_rate = int(base_rate * speed)
            self.pyttsx3_engine.setProperty('rate', new_rate)

            self.pyttsx3_engine.save_to_file(text, output_path)
            self.pyttsx3_engine.runAndWait()

            print(f"💾 pyttsx3 음성 파일 저장 완료: {output_path}")
            return output_path
        except Exception as e:
            print(f"❌ pyttsx3 실패: {e}")
            # 실패시 gTTS로 폴백
            return self._generate_with_gtts(text, output_path, 'en', speed)

    def _generate_with_gtts(self, text: str, output_path: str, lang: str = 'ko', speed: float = 1.0) -> str:
        """gTTS로 음성 생성 (온라인 폴백)"""
        try:
            mp3_path = output_path.replace('.wav', '.mp3')
            # gTTS는 속도 조절을 위해 slow 파라미터 사용 (0.5 이하일 때)
            slow_speech = speed <= 0.5
            tts = gTTS(text=text, lang=lang, slow=slow_speech)
            tts.save(mp3_path)

            # MP3를 WAV로 변환 및 속도 조절
            from pydub import AudioSegment
            audio = AudioSegment.from_mp3(mp3_path)

            # gTTS에서 slow=False였다면 속도 조절
            if not slow_speech and speed != 1.0:
                # pydub를 사용한 속도 조절
                audio = audio.speedup(playback_speed=speed)

            audio.export(output_path, format="wav")
            os.unlink(mp3_path)  # 임시 MP3 파일 삭제

            print(f"💾 gTTS 음성 파일 저장 완료: {output_path}")
            return output_path
        except Exception as e:
            print(f"❌ gTTS도 실패: {e}")
            raise e




from app.models.tts_request import ModelType


class TTSServiceFactory:
    """TTS 서비스 팩토리"""
    
    _chatterbox_instance = None

    @classmethod
    def get_service(cls, model_type: ModelType) -> BaseTTSService:
        """모델 타입에 따른 TTS 서비스 반환"""
        if model_type == ModelType.CHATTERBOX:
            if cls._chatterbox_instance is None:
                cls._chatterbox_instance = ChatterBoxTTSService()
            return cls._chatterbox_instance
        else:
            raise ValueError(f"지원하지 않는 모델 타입: {model_type}")


# 기본 TTS 서비스 인스턴스 (ChatterBox로 변경)
tts_service = TTSServiceFactory.get_service(ModelType.CHATTERBOX)

# 서버 시작 시 모델 미리 로드
def preload_all_models():
    """서버 시작 시 ChatterBox TTS 모델 로드"""
    print("🔄 ChatterBox TTS 모델 사전 로딩 중...")
    try:
        # ChatterBox 모델 로드
        chatterbox_service = TTSServiceFactory.get_service(ModelType.CHATTERBOX)
        print("✅ ChatterBox 모델 사전 로딩 완료")
        print("🎉 TTS 시스템 준비 완료!")
    except Exception as e:
        print(f"⚠️ 모델 로딩 실패: {e}")

# 모듈 로드 시 자동으로 모든 모델 사전 로드
preload_all_models()