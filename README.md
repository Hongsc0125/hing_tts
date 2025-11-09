# Modern TTS API - Multi-Engine Edition 🎙️

현대적 Text-to-Speech REST API 서비스로, 다중 TTS 엔진을 지원하는 실용적인 포트폴리오 프로젝트입니다.

## ✨ 주요 특징

### 🔄 **지능형 다중 엔진 시스템**
- **ChatterBox TTS**: 실제 TTS 라이브러리 기반의 안정적인 구현
- **Advanced ZONOS**: AI 기반 고급 기능 지원
- **자동 폴백**: 엔진 실패 시 자동으로 다른 엔진으로 전환

### 🌍 **다국어 지원**
- 🇰🇷 **한국어**: Microsoft Edge TTS로 고품질 음성 생성
- 🇺🇸 **영어**: 로컬 pyttsx3 엔진으로 빠른 처리
- 🌐 **다국어**: Google TTS를 백업으로 활용

### 🎯 **실용적 기능**
- RESTful API 설계
- 실시간 음성 생성
- 배치 처리 지원
- 음성 복제 기능
- 감정 제어 (고급 모델)

## 🚀 빠른 시작

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. 서버 실행
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 3. API 테스트
```bash
# 한국어 TTS 테스트
curl -X POST "http://localhost:8000/api/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "안녕하세요, 한국어 TTS 테스트입니다.",
    "model_type": "chatterbox"
  }' \
  --output test_korean.wav

# 영어 TTS 테스트
curl -X POST "http://localhost:8000/api/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is English TTS test.",
    "model_type": "chatterbox"
  }' \
  --output test_english.wav
```

## 📚 API 문서

서버 실행 후 다음 URL에서 Swagger UI로 API 문서를 확인할 수 있습니다:
- http://localhost:8000/docs

## 🏗️ 프로젝트 구조

```
app/
├── main.py              # FastAPI 애플리케이션 진입점
├── api/                 # API 엔드포인트
│   ├── tts.py          # 표준 TTS API
│   └── advanced_zonos.py # 고급 ZONOS API
├── services/            # 비즈니스 로직
│   ├── tts_service.py  # 메인 TTS 서비스
│   ├── zonos_tts_service.py
│   └── advanced_zonos_tts_service.py
└── models/              # 데이터 모델
    └── tts_request.py
```

## 🎨 지원 모델

### ChatterBox TTS (기본 모델)
- **구성**: Edge-TTS + pyttsx3 + gTTS
- **장점**: 안정성, 빠른 속도, 다국어 지원
- **한국어**: Microsoft Edge TTS (ko-KR-SunHiNeural)
- **영어**: 로컬 pyttsx3 엔진
- **백업**: Google TTS

### Advanced ZONOS TTS (고급 모델)
- **AI 기반**: 딥러닝 모델 활용
- **감정 제어**: 7가지 프리셋 + 커스텀
- **음성 복제**: 샘플 기반 화자 복제
- **배치 처리**: 다중 텍스트 동시 처리

## 🔧 주요 API 엔드포인트

### 기본 API (`/api/v1/`)
- `POST /generate` - 음성 생성
- `GET /voices` - 사용 가능한 음성 목록
- `GET /models` - 지원 모델 목록

### 고급 API (`/api/tts/advanced/`)
- `POST /generate` - 고급 음성 생성 (감정, 화자 제어)
- `POST /batch` - 배치 처리
- `POST /speaker-embedding` - 음성 복제
- `GET /emotions` - 감정 프리셋 목록

## 💡 기술적 특징

### 아키텍처
- **FastAPI**: 현대적 Python 웹 프레임워크
- **Factory Pattern**: 다중 TTS 엔진 관리
- **Dependency Injection**: 모듈화된 서비스 구조
- **Error Handling**: 견고한 예외 처리

### 성능 최적화
- **지연 로딩**: 필요시에만 모델 로드
- **캐싱**: 반복 요청 최적화
- **비동기 처리**: FastAPI 비동기 지원
- **스트리밍**: 대용량 오디오 처리

### 코드 품질
- **타입 힌팅**: Python 타입 시스템 활용
- **Pydantic**: 데이터 검증 및 직렬화
- **모듈화**: 관심사 분리
- **문서화**: 자동 API 문서 생성

## 🎯 포트폴리오 하이라이트

이 프로젝트는 다음과 같은 실무 스킬을 보여줍니다:

1. **웹 API 개발**: FastAPI를 활용한 RESTful API 설계
2. **다중 시스템 통합**: 여러 TTS 엔진의 통합 관리
3. **오류 처리**: 폴백 메커니즘과 예외 처리
4. **성능 최적화**: 캐싱과 비동기 처리
5. **코드 품질**: 타입 힌팅과 모듈화
6. **실용성**: 실제 사용 가능한 완성된 서비스

## 🌟 데모 및 활용

- **한국어 지원**: Microsoft Edge TTS로 자연스러운 한국어 음성
- **실시간 처리**: 빠른 응답 시간
- **확장성**: 새로운 TTS 엔진 추가 가능
- **안정성**: 다중 폴백 시스템

## 📄 라이센스

MIT License

## 👨‍💻 개발자

포트폴리오 프로젝트로 개발된 현대적 TTS API 서비스입니다.