import os
import tempfile
from pathlib import Path
from TTS.api import TTS


class TTSService:
    def __init__(self):
        self.model_path = "/home/hsc0125/Hing_tts/models/xtts-v2"
        self.tts = None
    
    def _initialize_model(self):
        """XTTS-v2 모델 초기화 (lazy loading)"""
        if self.tts is not None:
            return
            
        try:
            # 로컬 모델 경로 사용
            config_path = os.path.join(self.model_path, "config.json")
            model_file = os.path.join(self.model_path, "model.pth")
            
            if os.path.exists(config_path) and os.path.exists(model_file):
                print("TTS 모델 로딩 중...")
                self.tts = TTS(model_path=self.model_path, config_path=config_path, gpu=False)
                print("로컬 TTS 모델 초기화 성공")
            else:
                print("로컬 모델 파일을 찾을 수 없습니다")
                raise FileNotFoundError("로컬 모델 파일이 없습니다")
        except Exception as e:
            print(f"TTS 모델 초기화 오류: {e}")
            raise e
    
    def generate_speech(self, text: str, speaker_wav: str = None) -> str:
        """
        텍스트로부터 음성 생성 (한국어 전용)
        
        Args:
            text: 음성으로 변환할 텍스트
            speaker_wav: 스피커 참조 오디오 경로 (선택사항)
            
        Returns:
            생성된 오디오 파일 경로
        """
        # 모델이 로딩되지 않았다면 초기화
        self._initialize_model()
        
        # 임시 출력 파일 생성
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            output_path = tmp_file.name
        
        # 참조 오디오가 없으면 기본 한국어 스피커 사용
        if not speaker_wav:
            # 중국어 샘플을 한국어 기본 스피커로 사용
            samples_dir = Path(self.model_path) / "samples"
            speaker_wav = str(samples_dir / "zh-cn-sample.wav")
        
        try:
            self.tts.tts_to_file(
                text=text,
                file_path=output_path,
                speaker_wav=speaker_wav,
                language="ko"
            )
            return output_path
        except Exception as e:
            # 오류 시 정리
            if os.path.exists(output_path):
                os.unlink(output_path)
            raise e


# 전역 TTS 서비스 인스턴스
tts_service = TTSService()