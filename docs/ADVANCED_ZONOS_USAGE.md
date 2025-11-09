# Advanced ZONOS TTS 사용법

## 개요

Advanced ZONOS TTS는 ZONOS v0.1 모델을 기반으로 한국어에 최적화된 고품질 음성 합성 서비스입니다.

## 주요 특징

### 🎯 한국어 최적화
- **언어 지원**: 한국어(`ko`) 또는 영어(`en-us`) 지원
- **최적화된 설정**: 한국어 억양과 발음에 맞춘 파라미터
- **자연스러운 속도**: 13 phonemes/초로 설정

### 🎭 감정 제어
- **8가지 프리셋**: neutral, happy, sad, angry, surprised, calm, expressive
- **커스텀 감정**: 8차원 벡터로 세밀한 감정 조절 가능
- **자연스러운 표현**: 감정과 pitch variation 연동

### 🎤 고품질 Voice Cloning
- **ResNet293 기반**: 128차원 speaker embedding
- **캐싱 시스템**: 동일 화자 재사용시 빠른 처리
- **노이즈 제거**: speaker_noised 옵션으로 품질 향상

### 📊 고급 기능
- **배치 처리**: 여러 텍스트 동시 생성
- **비동기 처리**: async/await 지원
- **시간 예측**: 생성 시간 사전 계산
- **44kHz 출력**: 고품질 오디오

## API 엔드포인트

### 1. 기본 음성 생성
```bash
POST /api/tts/advanced/generate
```

**요청 예제:**
```json
{
  "text": "안녕하세요! Advanced ZONOS TTS 서비스입니다.",
  "speaker_name": "default",
  "emotion": "neutral",
  "cfg_scale": 2.5,
  "model_type": "transformer"
}
```

### 2. 감정 제어 음성 생성
```json
{
  "text": "오늘 정말 기분이 좋네요!",
  "speaker_name": "한국여성1_차분한",
  "emotion": "happy",
  "cfg_scale": 3.0
}
```

### 3. 커스텀 감정 벡터
```json
{
  "text": "이것은 매우 표현력이 풍부한 음성입니다.",
  "emotion": [0.25, 0.1, 0.05, 0.1, 0.15, 0.1, 0.2, 0.05],
  "custom_settings": {
    "pitch_std": 40.0,
    "speaking_rate": 11.0
  }
}
```

### 4. 배치 생성
```bash
POST /api/tts/advanced/batch
```

```json
{
  "texts": [
    "첫 번째 문장입니다.",
    "두 번째 문장입니다.",
    "세 번째 문장입니다."
  ],
  "speaker_names": ["한국여성1_차분한", "한국남성1_중후한", "한국여성2_밝은"],
  "emotions": ["neutral", "calm", "happy"]
}
```

## 최적 설정값

### 한국어 기본 설정
```python
korean_optimal_settings = {
    "language": "ko",                    # 한국어
    "fmax": 22050.0,                    # Voice cloning 최적값
    "pitch_std": 30.0,                  # 적당한 억양
    "speaking_rate": 13.0,              # 자연스러운 속도
    "cfg_scale": 2.5,                   # 안정적 생성
    "emotion": [0.15, 0.05, 0.05, 0.05, 0.05, 0.05, 0.2, 0.4]  # 중성적
}
```

### 감정별 권장 설정

#### 1. 뉴스/나레이션 (중성적)
- **emotion**: "neutral" 또는 [0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.55]
- **pitch_std**: 20.0-25.0
- **speaking_rate**: 14.0-16.0

#### 2. 활발한/밝은 (기쁨)
- **emotion**: "happy" 또는 [0.6, 0.05, 0.05, 0.05, 0.1, 0.05, 0.1, 0.05]
- **pitch_std**: 35.0-45.0
- **speaking_rate**: 15.0-18.0

#### 3. 차분한/진중한 (차분)
- **emotion**: "calm" 또는 [0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.55]
- **pitch_std**: 20.0-30.0
- **speaking_rate**: 11.0-13.0

#### 4. 감정 표현 (표현력 있는)
- **emotion**: "expressive" 또는 [0.25, 0.1, 0.05, 0.1, 0.15, 0.1, 0.2, 0.05]
- **pitch_std**: 40.0-60.0
- **speaking_rate**: 12.0-15.0

## Voice Cloning 가이드

### 1. 오디오 샘플 준비
- **길이**: 10-30초 권장 (최소 3초)
- **품질**: 고품질 WAV 또는 FLAC 형식
- **내용**: 깨끗한 음성만 포함
- **배경음**: 없거나 최소화

### 2. Speaker Embedding 생성
```bash
POST /api/tts/advanced/speaker-embedding
```

```json
{
  "audio_file_path": "/path/to/speaker_sample.wav"
}
```

### 3. 노이즈가 있는 샘플 처리
```json
{
  "custom_settings": {
    "speaker_noised": true,
    "dnsmos_ovrl": 4.0,
    "vqscore_8": [0.78, 0.78, 0.78, 0.78, 0.78, 0.78, 0.78, 0.78]
  }
}
```

## 성능 최적화

### 1. 캐싱 활용
- Speaker embedding은 자동으로 캐시됨
- 동일 화자 재사용시 빠른 처리
- 필요시 캐시 수동 클리어 가능

### 2. 모델 선택
- **Transformer**: 빠른 처리, 기본 기능
- **Hybrid**: 고급 기능, 더 많은 리소스 필요

### 3. 배치 처리
- 여러 문장 동시 처리로 효율성 향상
- 각 문장마다 다른 화자/감정 적용 가능

## 문제 해결

### 1. 한국어 지원 문제
```
⚠️ 경고: 한국어('ko')가 공식 지원 언어 목록에 없습니다.
🔄 대체 언어로 'en-us' 사용을 권장합니다.
```
→ 시스템이 자동으로 'en-us'로 대체합니다.

### 2. 메모리 부족
- Hybrid 모델 대신 Transformer 모델 사용
- 배치 크기 줄이기
- 캐시 주기적 클리어

### 3. 품질 개선
- 고품질 speaker 샘플 사용
- speaker_noised=True 설정
- cfg_scale 조정 (2.0-3.5)

## 실제 사용 예제

### Python 클라이언트
```python
import requests
import json

# 기본 생성
response = requests.post("http://localhost:8000/api/tts/advanced/generate", 
    json={
        "text": "안녕하세요, Advanced ZONOS TTS입니다.",
        "speaker_name": "default",
        "emotion": "neutral"
    })

with open("output.wav", "wb") as f:
    f.write(response.content)

# 감정 제어
response = requests.post("http://localhost:8000/api/tts/advanced/generate",
    json={
        "text": "와! 정말 놀라운 기술이네요!",
        "emotion": "surprised",
        "cfg_scale": 3.0
    })

# 배치 처리
response = requests.post("http://localhost:8000/api/tts/advanced/batch",
    json={
        "texts": ["첫 번째", "두 번째", "세 번째"],
        "emotions": ["neutral", "happy", "calm"]
    })

batch_results = response.json()
```

### cURL 예제
```bash
# 기본 생성
curl -X POST "http://localhost:8000/api/tts/advanced/generate" \
  -H "Content-Type: application/json" \
  -d '{"text": "안녕하세요!", "emotion": "happy"}' \
  --output output.wav

# 모델 정보 확인
curl "http://localhost:8000/api/tts/advanced/info?model_type=transformer"

# 감정 목록
curl "http://localhost:8000/api/tts/advanced/emotions"
```

## 성능 벤치마크

### 생성 속도 (RTX 4090 기준)
- **Transformer**: ~2x real-time (2초 오디오 → 1초 처리)
- **Hybrid**: ~1.5x real-time (더 높은 품질)

### 메모리 사용량
- **Transformer**: ~6GB VRAM
- **Hybrid**: ~8GB VRAM + 3000시리즈 이상 GPU 필요

### 품질 지표
- **샘플링 레이트**: 44.1kHz
- **비트 깊이**: 16-bit
- **지연시간**: <2초 (cold start), <0.5초 (warm)

---

이 문서는 Advanced ZONOS TTS의 완전한 사용법을 다룹니다. 추가 질문이나 문제가 있으면 이슈를 생성해주세요.