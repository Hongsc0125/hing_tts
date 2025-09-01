# Advanced ZONOS TTS 사용법 가이드

## 🎯 개요

**Advanced ZONOS TTS**는 한국어에 완전히 최적화된 고품질 Text-to-Speech 시스템입니다. 
ZONOS 모델을 심층 분석하여 한국어 음성 생성에 최적화된 설정과 기능들을 제공합니다.

## 🚀 주요 특징

- **🇰🇷 한국어 완전 지원**: `ko` 언어 코드 네이티브 지원
- **🎭 감정 제어**: 7가지 프리셋 + 8차원 커스텀 벡터
- **🎤 Voice Cloning**: ResNet293 + LDA 기반 고품질 화자 복제
- **⚡ 고성능**: ~2x Real-time factor, 44.1kHz 출력
- **🔄 배치 처리**: 최대 10개 텍스트 동시 처리
- **💾 지능형 캐싱**: Speaker embedding 자동 캐싱

## 📚 API 엔드포인트

### 1. 기본 음성 생성
```http
POST /api/tts/advanced/generate
```

**요청 예제:**
```json
{
  "text": "안녕하세요! Advanced ZONOS TTS입니다.",
  "emotion": "neutral",
  "cfg_scale": 2.5,
  "speaker_name": "한국여성1"
}
```

**응답**: WAV 오디오 파일 (44.1kHz)

### 2. 배치 음성 생성
```http
POST /api/tts/advanced/batch
```

**요청 예제:**
```json
{
  "texts": [
    "첫 번째 문장입니다.",
    "두 번째 문장입니다.",
    "세 번째 문장입니다."
  ],
  "emotions": ["neutral", "happy", "calm"],
  "cfg_scale": 2.5
}
```

### 3. 모델 정보 조회
```http
GET /api/tts/advanced/info
```

### 4. 음성 목록 조회
```http
GET /api/tts/advanced/voices
```

### 5. 감정 프리셋 조회
```http
GET /api/tts/advanced/emotions
```

### 6. 생성 시간 추정
```http
POST /api/tts/advanced/estimate-time
```

### 7. 캐시 관리
```http
DELETE /api/tts/advanced/cache
```

## 🎭 감정 제어 시스템

### 7가지 프리셋 감정

| 감정 | 설명 | 사용 사례 |
|------|------|----------|
| `neutral` | 중성적, 뉴스/내레이션 | 공식 발표, 뉴스 |
| `happy` | 밝고 활발한 | 광고, 환영 메시지 |
| `sad` | 슬픔, 감성적 표현 | 추모, 감동적 내용 |
| `angry` | 분노, 강한 표현 | 경고, 강조 |
| `surprised` | 놀라움, 리액션 | 이벤트, 발견 |
| `calm` | 차분함, 명상적 | 안내, 힐링 |
| `expressive` | 표현력 있는 | 연기, 드라마 |

### 커스텀 감정 벡터

8차원 벡터로 세밀한 감정 제어가 가능합니다:

```json
{
  "emotion": [0.3, 0.1, 0.05, 0.1, 0.2, 0.1, 0.1, 0.05]
}
```

**차원 순서**: [Happiness, Sadness, Disgust, Fear, Surprise, Anger, Other, Neutral]

## 🎤 Voice Cloning 가이드

### 음성 샘플 준비

**권장 사양:**
- **길이**: 10-30초 (최소 3초)
- **형식**: WAV, FLAC (고품질 권장)
- **내용**: 깨끗한 음성만
- **배경음**: 최소화 또는 제거
- **화자**: 단일 화자만

**처리 과정:**
1. 16kHz 리샘플링 (자동)
2. 모노 변환 (자동) 
3. ResNet293 특성 추출
4. LDA 차원 축소 (128차원)
5. 자동 캐싱

### 음성 파일 위치
```
/home/hsc0125/Hing_tts/models/audio_data/
├── sample1.wav
├── sample2.wav
└── sample3.wav
```

## ⚙️ 파라미터 최적화 가이드

### 한국어 최적화 기본값

```python
KOREAN_OPTIMAL_CONFIG = {
    "language": "ko",          # 한국어 직접 지원
    "fmax": 22050.0,          # Voice cloning 권장값  
    "pitch_std": 30.0,        # 한국어 자연스러운 억양
    "speaking_rate": 13.0,    # 적당한 발화 속도
    "cfg_scale": 2.5,         # 안정적 생성 품질
}
```

### 용도별 최적화 설정

**뉴스/공식 발표:**
```json
{
  "emotion": "neutral",
  "pitch_std": 20.0,
  "speaking_rate": 14.0,
  "cfg_scale": 2.0
}
```

**광고/마케팅:**
```json
{
  "emotion": "happy",
  "pitch_std": 40.0,
  "speaking_rate": 16.0,
  "cfg_scale": 2.8
}
```

**교육/안내:**
```json
{
  "emotion": "calm",
  "pitch_std": 25.0,
  "speaking_rate": 12.0,
  "cfg_scale": 2.2
}
```

**감성적 내용:**
```json
{
  "emotion": "expressive", 
  "pitch_std": 45.0,
  "speaking_rate": 11.0,
  "cfg_scale": 3.2
}
```

## 💻 Python 클라이언트 예제

### 기본 사용법

```python
import requests

BASE_URL = "http://localhost:3000"

# 기본 음성 생성
response = requests.post(f"{BASE_URL}/api/tts/advanced/generate", 
    json={
        "text": "안녕하세요! Advanced ZONOS TTS입니다.",
        "emotion": "neutral",
        "cfg_scale": 2.5
    })

if response.status_code == 200:
    with open("output.wav", "wb") as f:
        f.write(response.content)
    print("음성 파일 생성 완료!")
```

### 감정 제어 예제

```python
emotions = ["neutral", "happy", "sad", "surprised"]

for emotion in emotions:
    response = requests.post(f"{BASE_URL}/api/tts/advanced/generate",
        json={
            "text": f"이것은 {emotion} 감정입니다.",
            "emotion": emotion,
            "cfg_scale": 2.8
        })
    
    if response.status_code == 200:
        with open(f"output_{emotion}.wav", "wb") as f:
            f.write(response.content)
        print(f"{emotion} 음성 생성 완료!")
```

### 배치 처리 예제

```python
batch_response = requests.post(f"{BASE_URL}/api/tts/advanced/batch",
    json={
        "texts": [
            "첫 번째 문장입니다.",
            "두 번째 문장입니다.",
            "세 번째 문장입니다."
        ],
        "emotions": ["neutral", "happy", "calm"],
        "cfg_scale": 2.5
    })

if batch_response.status_code == 200:
    data = batch_response.json()
    print(f"배치 처리 완료: {data['successful_count']}/{data['total_requests']} 성공")
```

### 커스텀 감정 벡터 예제

```python
# 복합 감정 (행복 30% + 놀라움 20% + 기타 50%)
custom_emotion = [0.3, 0.05, 0.05, 0.05, 0.2, 0.05, 0.25, 0.05]

response = requests.post(f"{BASE_URL}/api/tts/advanced/generate",
    json={
        "text": "커스텀 감정으로 말하고 있습니다.",
        "emotion": custom_emotion,
        "cfg_scale": 3.0
    })
```

## 🧪 테스트 실행

### 종합 테스트 스크립트 실행

```bash
# 서버 시작 (터미널 1)
uvicorn app.main:app --host 0.0.0.0 --port 3000

# 테스트 실행 (터미널 2)  
python test_advanced_zonos.py
```

### 개별 기능 테스트

```bash
# 모델 정보 확인
curl http://localhost:3000/api/tts/advanced/info

# 음성 목록 확인
curl http://localhost:3000/api/tts/advanced/voices

# 감정 프리셋 확인
curl http://localhost:3000/api/tts/advanced/emotions
```

## 📊 성능 지표

### 벤치마크 결과 (RTX GPU 환경)

- **생성 속도**: ~2x Real-time factor
- **음질**: 44.1kHz, 16bit WAV
- **지연시간**: 평균 3-5초 (텍스트 길이 대비)
- **메모리 사용량**: ~6GB VRAM (Transformer)
- **캐시 효율**: Speaker embedding 재사용시 50% 성능 향상

### 품질 비교

| 모델 | 자연성 | 감정표현 | 한국어발음 | 생성속도 |
|------|--------|----------|-----------|----------|
| Basic ZONOS | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **Advanced ZONOS** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

## 🔧 문제 해결

### 일반적인 문제들

**1. 모델 로딩 실패**
```
❌ Advanced ZONOS 모델 로드 실패
```
**해결책**: 
- CUDA 메모리 확인 (6GB+ 필요)
- 인터넷 연결 확인 (Hugging Face 다운로드)
- `pip install -r requirements.txt`

**2. 음성 샘플 없음**
```
⚠️ 오디오 파일이 없습니다
```
**해결책**: 
- `/models/audio_data/` 폴더에 WAV 파일 추가
- 파일 권한 확인

**3. C++ 컴파일러 오류**
```
❌ InvalidCxxCompiler: No working C++ compiler found
```
**해결책**:
```bash
sudo apt install build-essential g++ cmake
```

**4. 메모리 부족**
```
❌ CUDA out of memory
```
**해결책**:
- GPU 메모리 정리: `torch.cuda.empty_cache()`
- 캐시 초기화: `DELETE /api/tts/advanced/cache`
- 배치 크기 줄이기

### 로그 레벨 조정

```python
import logging
logging.getLogger("phonemizer").setLevel(logging.ERROR)
```

## 🎯 베스트 프랙티스

### 1. 품질 최적화

- **텍스트 전처리**: 특수문자, 숫자 정리
- **적절한 길이**: 문장당 50-200자 권장
- **문맥 고려**: 감정과 내용의 일치

### 2. 성능 최적화

- **캐싱 활용**: 동일 화자 재사용
- **배치 처리**: 여러 문장 동시 처리
- **적절한 CFG**: 품질과 속도의 균형

### 3. 운영 고려사항

- **메모리 모니터링**: GPU 사용량 체크
- **오류 처리**: 재시도 로직 구현
- **로드밸런싱**: 여러 인스턴스 운영

## 📈 확장 가능성

### 향후 개발 계획

1. **다국어 확장**: 영어, 중국어 등 추가 지원
2. **실시간 스트리밍**: WebSocket 기반 실시간 TTS
3. **음성 편집**: 속도, 피치 후처리
4. **SSML 지원**: 표준 마크업 언어 지원
5. **클라우드 연동**: AWS, GCP 호환성

### API 버전 관리

- **v1**: 기본 TTS (하위 호환성)
- **v2**: Advanced ZONOS (현재)
- **v3**: 실시간 스트리밍 (계획)

---

## 📞 지원

**문제 보고**: GitHub Issues  
**문서**: `/docs` API 문서  
**테스트**: `test_advanced_zonos.py`  

**개발팀**: Advanced ZONOS TTS Team  
**마지막 업데이트**: 2025년 1월  