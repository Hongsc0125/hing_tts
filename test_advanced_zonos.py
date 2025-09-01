#!/usr/bin/env python3
"""
Advanced ZONOS TTS 종합 테스트 스크립트
======================================

완전히 새로 구현된 Advanced ZONOS TTS의 모든 기능을 테스트합니다.
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, Any

BASE_URL = "http://localhost:3000"

async def test_api_endpoint(session: aiohttp.ClientSession, method: str, endpoint: str, data: Dict[Any, Any] = None) -> Dict:
    """API 엔드포인트 테스트"""
    url = f"{BASE_URL}{endpoint}"
    
    try:
        if method.upper() == "GET":
            async with session.get(url, params=data) as response:
                result = await response.json() if response.content_type == 'application/json' else None
                return {"status": response.status, "data": result}
        
        elif method.upper() == "POST":
            async with session.post(url, json=data) as response:
                if response.content_type == 'application/json':
                    result = await response.json()
                elif response.content_type.startswith('audio/'):
                    # 오디오 파일 저장
                    filename = f"test_output_{int(time.time())}.wav"
                    with open(filename, 'wb') as f:
                        f.write(await response.read())
                    result = {"audio_saved": filename, "headers": dict(response.headers)}
                else:
                    result = await response.text()
                
                return {"status": response.status, "data": result}
                
        elif method.upper() == "DELETE":
            async with session.delete(url) as response:
                result = await response.json() if response.content_type == 'application/json' else await response.text()
                return {"status": response.status, "data": result}
                
    except Exception as e:
        return {"status": "error", "error": str(e)}


async def run_comprehensive_tests():
    """종합 테스트 실행"""
    print("🚀 Advanced ZONOS TTS 종합 테스트 시작")
    print("=" * 60)
    
    async with aiohttp.ClientSession() as session:
        
        # 1. 모델 정보 조회
        print("📊 1. 모델 정보 조회 테스트")
        result = await test_api_endpoint(session, "GET", "/api/tts/advanced/info")
        if result["status"] == 200:
            print("✅ 모델 정보 조회 성공")
            print(f"   모델: {result['data'].get('model_name')}")
            print(f"   디바이스: {result['data'].get('device')}")
            print(f"   음성 수: {result['data'].get('korean_voices_count')}")
            print(f"   로드 상태: {result['data'].get('is_loaded')}")
        else:
            print(f"❌ 모델 정보 조회 실패: {result}")
        print()
        
        # 2. 음성 목록 조회
        print("🎤 2. 한국어 음성 목록 조회 테스트")
        result = await test_api_endpoint(session, "GET", "/api/tts/advanced/voices")
        if result["status"] == 200:
            print("✅ 음성 목록 조회 성공")
            voices = result['data'].get('voices', [])
            print(f"   사용 가능한 음성: {len(voices)}개")
            for voice in voices:
                print(f"     - {voice}")
        else:
            print(f"❌ 음성 목록 조회 실패: {result}")
        print()
        
        # 3. 감정 프리셋 조회
        print("🎭 3. 감정 프리셋 조회 테스트")
        result = await test_api_endpoint(session, "GET", "/api/tts/advanced/emotions")
        if result["status"] == 200:
            print("✅ 감정 프리셋 조회 성공")
            presets = result['data'].get('presets', {})
            print(f"   사용 가능한 감정: {len(presets)}개")
            for emotion in presets.keys():
                print(f"     - {emotion}")
        else:
            print(f"❌ 감정 프리셋 조회 실패: {result}")
        print()
        
        # 4. 시간 추정 테스트
        print("⏱️ 4. 생성 시간 추정 테스트")
        test_text = "안녕하세요! Advanced ZONOS TTS 시간 추정 테스트입니다."
        result = await test_api_endpoint(session, "POST", "/api/tts/advanced/estimate-time", test_text)
        if result["status"] == 200:
            print("✅ 시간 추정 성공")
            est_data = result['data']
            print(f"   텍스트 길이: {est_data.get('text_length')}자")
            print(f"   예상 음성 길이: {est_data.get('estimated_audio_duration'):.2f}초")
            print(f"   예상 생성 시간: {est_data.get('estimated_generation_time'):.2f}초")
        else:
            print(f"❌ 시간 추정 실패: {result}")
        print()
        
        # 5. 기본 음성 생성 테스트
        print("🎵 5. 기본 음성 생성 테스트")
        start_time = time.time()
        
        tts_request = {
            "text": "안녕하세요! Advanced ZONOS TTS 기본 음성 생성 테스트입니다.",
            "emotion": "neutral",
            "cfg_scale": 2.5
        }
        
        result = await test_api_endpoint(session, "POST", "/api/tts/advanced/generate", tts_request)
        generation_time = time.time() - start_time
        
        if result["status"] == 200:
            print("✅ 기본 음성 생성 성공")
            headers = result['data'].get('headers', {})
            print(f"   오디오 파일: {result['data'].get('audio_saved')}")
            print(f"   생성 시간: {generation_time:.2f}초")
            print(f"   음성 길이: {headers.get('X-Audio-Duration')}초")
            print(f"   샘플링 레이트: {headers.get('X-Sample-Rate')}Hz")
            print(f"   파일 크기: {int(headers.get('X-File-Size', 0))/1024:.1f}KB")
        else:
            print(f"❌ 기본 음성 생성 실패: {result}")
        print()
        
        # 6. 감정별 음성 생성 테스트
        emotions_to_test = ["happy", "sad", "surprised", "calm"]
        print(f"🎭 6. 감정별 음성 생성 테스트 ({len(emotions_to_test)}가지)")
        
        emotion_results = []
        for emotion in emotions_to_test:
            print(f"   🎭 {emotion} 감정 테스트 중...")
            start_time = time.time()
            
            tts_request = {
                "text": f"이것은 {emotion} 감정으로 말하는 Advanced ZONOS TTS입니다.",
                "emotion": emotion,
                "cfg_scale": 2.8
            }
            
            result = await test_api_endpoint(session, "POST", "/api/tts/advanced/generate", tts_request)
            generation_time = time.time() - start_time
            
            if result["status"] == 200:
                print(f"     ✅ {emotion} 성공 ({generation_time:.2f}초)")
                emotion_results.append((emotion, True, generation_time))
            else:
                print(f"     ❌ {emotion} 실패")
                emotion_results.append((emotion, False, generation_time))
        
        successful_emotions = sum(1 for _, success, _ in emotion_results if success)
        avg_time = sum(t for _, success, t in emotion_results if success) / max(successful_emotions, 1)
        print(f"   📊 감정 테스트 결과: {successful_emotions}/{len(emotions_to_test)} 성공, 평균 {avg_time:.2f}초")
        print()
        
        # 7. 커스텀 감정 벡터 테스트
        print("🎨 7. 커스텀 감정 벡터 테스트")
        start_time = time.time()
        
        custom_emotion = [0.3, 0.1, 0.05, 0.1, 0.2, 0.1, 0.1, 0.05]  # 복합 감정
        tts_request = {
            "text": "커스텀 감정 벡터로 생성된 Advanced ZONOS TTS 음성입니다.",
            "emotion": custom_emotion,
            "cfg_scale": 3.0
        }
        
        result = await test_api_endpoint(session, "POST", "/api/tts/advanced/generate", tts_request)
        generation_time = time.time() - start_time
        
        if result["status"] == 200:
            print("✅ 커스텀 감정 벡터 테스트 성공")
            print(f"   생성 시간: {generation_time:.2f}초")
            print(f"   오디오 파일: {result['data'].get('audio_saved')}")
        else:
            print(f"❌ 커스텀 감정 벡터 테스트 실패: {result}")
        print()
        
        # 8. 배치 처리 테스트
        print("🔄 8. 배치 처리 테스트")
        start_time = time.time()
        
        batch_request = {
            "texts": [
                "첫 번째 배치 테스트 문장입니다.",
                "두 번째 배치 테스트 문장입니다.", 
                "세 번째 배치 테스트 문장입니다."
            ],
            "emotions": ["neutral", "happy", "calm"],
            "cfg_scale": 2.5
        }
        
        result = await test_api_endpoint(session, "POST", "/api/tts/advanced/batch", batch_request)
        batch_time = time.time() - start_time
        
        if result["status"] == 200:
            print("✅ 배치 처리 테스트 성공")
            batch_data = result['data']
            print(f"   처리된 텍스트: {batch_data.get('total_requests')}개")
            print(f"   성공한 생성: {batch_data.get('successful_count')}개") 
            print(f"   총 처리 시간: {batch_time:.2f}초")
        else:
            print(f"❌ 배치 처리 테스트 실패: {result}")
        print()
        
        # 9. 캐시 테스트
        print("🔄 9. 캐시 시스템 테스트")
        
        # 캐시 초기화
        result = await test_api_endpoint(session, "DELETE", "/api/tts/advanced/cache")
        if result["status"] == 200:
            print("✅ 캐시 초기화 성공")
        else:
            print(f"❌ 캐시 초기화 실패: {result}")
        print()
        
        # 10. 종합 성능 테스트
        print("⚡ 10. 종합 성능 테스트")
        
        performance_texts = [
            "짧은 문장 테스트",
            "중간 길이의 문장으로 성능을 테스트해보겠습니다.",
            "이것은 긴 문장으로 Advanced ZONOS TTS의 성능과 품질을 종합적으로 평가하기 위한 테스트 문장입니다."
        ]
        
        total_generation_time = 0
        total_audio_duration = 0
        successful_tests = 0
        
        for i, text in enumerate(performance_texts):
            print(f"   📝 성능 테스트 {i+1}/{len(performance_texts)}")
            start_time = time.time()
            
            tts_request = {
                "text": text,
                "emotion": "neutral", 
                "cfg_scale": 2.5
            }
            
            result = await test_api_endpoint(session, "POST", "/api/tts/advanced/generate", tts_request)
            generation_time = time.time() - start_time
            
            if result["status"] == 200:
                headers = result['data'].get('headers', {})
                audio_duration = float(headers.get('X-Audio-Duration', 0))
                
                total_generation_time += generation_time
                total_audio_duration += audio_duration
                successful_tests += 1
                
                print(f"     ✅ 성공 - 생성시간: {generation_time:.2f}초, 음성길이: {audio_duration:.2f}초")
            else:
                print(f"     ❌ 실패")
        
        if successful_tests > 0:
            avg_rtf = total_audio_duration / total_generation_time  # Real-time factor
            print(f"   📊 성능 요약:")
            print(f"     성공률: {successful_tests}/{len(performance_texts)} ({successful_tests/len(performance_texts)*100:.1f}%)")
            print(f"     평균 생성 시간: {total_generation_time/successful_tests:.2f}초")
            print(f"     평균 음성 길이: {total_audio_duration/successful_tests:.2f}초")
            print(f"     Real-time factor: {avg_rtf:.1f}x")
        
        print()
        print("🎉 Advanced ZONOS TTS 종합 테스트 완료!")
        print("=" * 60)


if __name__ == "__main__":
    print("🧪 Advanced ZONOS TTS 테스트 스크립트")
    print("서버가 http://localhost:3000 에서 실행 중인지 확인하세요.")
    
    try:
        asyncio.run(run_comprehensive_tests())
    except KeyboardInterrupt:
        print("\n⏹️  테스트가 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 테스트 실행 중 오류 발생: {e}")