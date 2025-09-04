# Finance Recommendation System

금융 거래 데이터를 활용한 추천 시스템 구현 프로젝트입니다.

## 🎯 프로젝트 개요

이 프로젝트는 신용카드 거래 데이터를 기반으로 사용자에게 상점을 추천하는 시스템을 구현합니다. 협업 필터링과 행렬 분해 기법을 사용하여 개인화된 추천을 제공합니다.

## 📊 데이터셋

- **거래 데이터**: 150만+ 거래 기록
- **사용자 데이터**: 2,000명의 사용자 정보
- **카드 데이터**: 6,000+ 카드 정보
- **상점 데이터**: MCC 코드 기반 카테고리 분류
- **사기 라벨**: 거래별 사기 여부 정보

## 🏗️ 시스템 아키텍처

```
finance_recommendation/
├── data/                           # 원본 데이터
│   ├── transactions_data.csv       # 거래 데이터
│   ├── users_data.csv              # 사용자 데이터
│   ├── cards_data.csv              # 카드 데이터
│   ├── mcc_codes.json              # 상점 카테고리 코드
│   └── train_fraud_labels.json     # 사기 라벨
├── src/                            # 소스 코드
│   ├── day1_data_preprocessing.py  # 데이터 전처리
│   ├── day1_quick_test.py          # 빠른 테스트
│   ├── day2_collaborative_filtering.py # 협업 필터링
│   ├── day3_matrix_factorization.py   # 행렬 분해
│   └── day4_evaluation.py          # 모델 평가
├── reports/                        # 분석 결과
└── notebooks/                      # Jupyter 노트북
```

## 🚀 구현된 추천 알고리즘

### 1. 협업 필터링 (Collaborative Filtering)
- **사용자 기반**: 유사한 사용자의 선호도를 기반으로 추천
- **아이템 기반**: 유사한 상점의 패턴을 기반으로 추천
- **유사도 측정**: 코사인 유사도 사용

### 2. 행렬 분해 (Matrix Factorization)
- **SVD (Singular Value Decomposition)**: 특잇값 분해를 통한 차원 축소
- **NMF (Non-negative Matrix Factorization)**: 음이 아닌 행렬 분해
- **잠재 요인**: 사용자와 상점의 숨겨진 특성 학습

## 📈 성능 평가

### 평가 지표
- **RMSE** (Root Mean Square Error): 평점 예측 정확도
- **MAE** (Mean Absolute Error): 평균 절댓값 오차
- **Precision@K**: 상위 K개 추천의 정밀도
- **Recall@K**: 상위 K개 추천의 재현율
- **F1-Score@K**: 정밀도와 재현율의 조화평균
- **Coverage**: 추천 다양성

### 실험 결과

| 모델 | RMSE | MAE | Precision@5 | Recall@5 | F1@5 |
|------|------|-----|-------------|----------|------|
| **협업 필터링** | N/A | N/A | 0.0000 | 0.0000 | 0.0000 |
| **SVD** | 0.2444 | 0.1644 | 0.0933 | 0.1911 | 0.1254 |
| **NMF** | 0.2444 | 0.1644 | 0.1241 | 0.2759 | 0.1712 |

> **결과 분석**: NMF 모델이 대부분의 랭킹 기반 메트릭에서 우수한 성능을 보였습니다.

## 💡 주요 특징

### 데이터 전처리
- 거래 금액 정규화 및 이상치 처리
- 시간 기반 특성 추출 (시간, 요일, 월)
- MCC 코드 기반 상점 카테고리 매핑
- 사기 거래 라벨 통합

### 추천 시스템
- **개인화된 추천**: 사용자별 거래 패턴 학습
- **실시간 추천**: 효율적인 행렬 연산
- **다양한 알고리즘**: 협업 필터링 + 행렬 분해
- **성능 최적화**: 희소 행렬 및 샘플링 활용

### 평가 시스템
- **교차 검증**: Train/Test 분할을 통한 검증
- **다중 메트릭**: 예측 정확도 + 랭킹 품질
- **모델 비교**: 알고리즘별 성능 비교 분석

## 🔧 사용법

### 환경 설정
```bash
# 필요한 패키지 설치
pip install pandas numpy scikit-learn matplotlib

# 프로젝트 클론
git clone <repository-url>
cd finance_recommendation
```

### 실행 방법

1. **데이터 전처리**
```bash
python src/day1_data_preprocessing.py
```

2. **협업 필터링 실행**
```bash
python src/day2_collaborative_filtering.py
```

3. **행렬 분해 모델 실행**
```bash
python src/day3_matrix_factorization.py
```

4. **성능 평가 실행**
```bash
python src/day4_evaluation.py
```

## 📋 프로젝트 일정

- **Day 1**: 데이터셋 로딩 및 전처리 ✅
- **Day 2**: 사용자/아이템 기반 추천 시스템 구현 ✅
- **Day 3**: Matrix Factorization/SVD 기반 추천 모델 ✅
- **Day 4**: 추천 결과 평가 (RMSE, Precision@k, Recall@k) ✅
- **Day 5**: 프로젝트 코드 정리 및 문서화 ✅
- **Weekend**: 프로젝트 리포트 작성 및 시각화 추가

## 🎓 학습 목표

1. **추천 시스템 이론 이해**: 협업 필터링과 행렬 분해의 원리
2. **실전 구현 경험**: Python을 활용한 추천 시스템 구축
3. **성능 평가 방법론**: 다양한 평가 지표 활용
4. **데이터 처리 파이프라인**: 대용량 데이터 전처리 및 최적화

## 🔍 추가 개선 방안

1. **하이브리드 모델**: 협업 필터링 + 콘텐츠 기반 필터링 결합
2. **딥러닝 모델**: Neural Collaborative Filtering, AutoEncoder 적용
3. **시계열 분석**: 시간 패턴을 고려한 동적 추천
4. **지리적 정보**: 사용자 위치 기반 상점 추천
5. **A/B 테스트**: 실제 환경에서의 추천 성능 검증

## 🏆 성과

- **다양한 추천 알고리즘 구현**: 3가지 접근법 비교 분석
- **체계적인 평가 프레임워크**: 종합적 성능 측정
- **실무 적용 가능한 코드**: 모듈화된 구조와 확장성
- **상세한 문서화**: 재현 가능한 프로젝트

## 📞 문의

프로젝트에 대한 질문이나 개선 제안이 있으시면 이슈를 등록해 주세요.

---
*이 프로젝트는 교육 목적으로 제작되었으며, 실제 금융 서비스에서는 추가적인 보안과 규정 준수가 필요합니다.*