# 🚀 Finance Recommendation System - Quick Start Guide

이 문서는 금융 추천 시스템 프로젝트를 빠르게 시작할 수 있는 가이드입니다.

## ⚡ 빠른 실행

### 1단계: 환경 설정
```bash
# 프로젝트 디렉토리로 이동
cd finance_recommendation

# 필요한 패키지 설치
pip install -r requirements.txt
```

### 2단계: 전체 파이프라인 실행
```bash
# 모든 단계 한 번에 실행
python src/main.py --step all --sample_size 25000

# 또는 개별 단계 실행
python src/main.py --step 1  # 데이터 전처리만
python src/main.py --step 2  # 협업 필터링만
python src/main.py --step 3  # 행렬 분해만
python src/main.py --step 4  # 평가만
```

### 3단계: 시각화 생성
```bash
# 종합 시각화 및 리포트
python src/weekend_visualization.py
```

## 📁 프로젝트 구조

```
finance_recommendation/
├── 📊 data/                     # 원본 데이터 (5개 파일)
├── 🔬 src/                      # 소스 코드
│   ├── main.py                  # 통합 실행 스크립트
│   ├── day1_data_preprocessing.py
│   ├── day1_quick_test.py
│   ├── day2_collaborative_filtering.py
│   ├── day3_matrix_factorization.py
│   ├── day4_evaluation.py
│   └── weekend_visualization.py
├── 📈 reports/                  # 결과 및 리포트
│   ├── project_report.md        # 📋 종합 프로젝트 보고서
│   ├── evaluation_results.json  # 📊 모델 평가 결과
│   ├── model_comparison.csv     # 📉 모델 성능 비교
│   └── *.png                    # 🎨 시각화 차트들
├── 📚 notebooks/                # Jupyter 노트북 (선택사항)
├── 📖 README.md                 # 프로젝트 상세 문서
├── 🚀 QUICKSTART.md            # 이 파일
└── 📦 requirements.txt          # 패키지 의존성
```

## 🎯 단계별 실행 가이드

### Day 1: 데이터 전처리
```bash
python src/day1_quick_test.py
```
- 📥 **입력**: 원본 CSV/JSON 데이터
- 📤 **출력**: 전처리된 데이터, 기본 통계
- ⏱️ **소요시간**: ~30초

### Day 2: 협업 필터링
```bash
python src/day2_collaborative_filtering.py
```
- 🤝 **구현**: 사용자/아이템 기반 협업 필터링
- 📊 **출력**: 유사도 행렬, 개인화 추천
- ⏱️ **소요시간**: ~2분

### Day 3: 행렬 분해
```bash
python src/day3_matrix_factorization.py
```
- 🔢 **구현**: SVD, NMF 모델
- 📈 **출력**: 잠재 요인 분석, 예측 평점
- ⏱️ **소요시간**: ~1분

### Day 4: 성능 평가
```bash
python src/day4_evaluation.py
```
- 📊 **평가**: RMSE, Precision@K, Recall@K
- 📋 **출력**: 모델 비교 리포트
- ⏱️ **소요시간**: ~3분

### Weekend: 시각화
```bash
python src/weekend_visualization.py
```
- 🎨 **생성**: 종합 대시보드, 분석 차트
- 🖼️ **출력**: PNG 시각화 파일들
- ⏱️ **소요시간**: ~1분

## 🔧 주요 매개변수

### 샘플 크기 조정
```bash
# 빠른 테스트용 (권장: 학습/개발)
python src/main.py --sample_size 10000

# 표준 크기 (권장: 평가)
python src/main.py --sample_size 30000

# 대용량 (주의: 메모리 부족 가능)
python src/main.py --sample_size 100000
```

### 모델 파라미터
- **SVD 컴포넌트**: 기본 20개 (day3_matrix_factorization.py 수정)
- **추천 개수**: 기본 5-10개 (각 스크립트에서 n_recommendations 변경)
- **평가 K값**: [5, 10, 20] (day4_evaluation.py에서 수정)

## 📊 결과 해석

### 성능 메트릭 이해
- **RMSE/MAE**: 낮을수록 좋음 (예측 정확도)
- **Precision@K**: 높을수록 좋음 (추천 정확도)
- **Recall@K**: 높을수록 좋음 (놓친 아이템 최소화)
- **F1@K**: 높을수록 좋음 (정확도와 재현율의 균형)

### 예상 성능 (30K 샘플 기준)
```
NMF 모델 (최고 성능):
- RMSE: ~0.24
- Precision@5: ~0.12
- Recall@5: ~0.28
- F1@5: ~0.17
```

## ⚠️ 문제 해결

### 메모리 부족 오류
```bash
# 샘플 크기 줄이기
python src/main.py --sample_size 15000
```

### 패키지 누락 오류
```bash
# 필수 패키지 재설치
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 파일 경로 오류
```bash
# 올바른 디렉토리에서 실행 확인
pwd  # /Users/.../finance_recommendation 이어야 함
ls data/  # 데이터 파일들이 있어야 함
```

### 성능이 너무 낮을 때
1. **데이터 크기 증가**: `--sample_size` 늘리기
2. **모델 파라미터 조정**: 잠재 요인 개수 증가
3. **피처 추가**: 시간/지리적 정보 활용

## 🔍 심화 분석

### 추가 분석을 위한 코드 수정
```python
# day3_matrix_factorization.py에서
mf.train_svd_model(n_components=50)  # 컴포넌트 증가

# day2_collaborative_filtering.py에서
cf.user_based_recommendations(user_id, n_recommendations=20)  # 더 많은 추천
```

### 새로운 평가 메트릭 추가
```python
# day4_evaluation.py에서 MAP, NDCG 등 추가 구현 가능
```

## 🎓 학습 가이드

### 초급자용 학습 순서
1. 📖 README.md 전체 읽기
2. 🚀 QUICKSTART.md로 실행 경험
3. 📋 project_report.md로 이론 학습
4. 💻 코드 리뷰 및 수정 실험

### 중급자용 확장 과제
1. 🔄 하이브리드 모델 구현
2. 🧠 딥러닝 모델 추가
3. ⏱️ 실시간 추천 시스템 구축
4. 📊 A/B 테스트 프레임워크 구현

### 고급자용 연구 주제
1. 🔐 연합학습 기반 추천
2. 🎯 강화학습 적용
3. 🔍 설명 가능한 추천
4. 🌐 멀티도메인 추천

## 📞 지원 및 기여

- **버그 리포트**: GitHub Issues 활용
- **기능 제안**: Pull Request 환영
- **질문**: README.md FAQ 섹션 참조

---

**즐거운 추천 시스템 개발 되세요! 🎉**

*최종 업데이트: 2024년 9월 3일*