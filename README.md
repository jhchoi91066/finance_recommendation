# 금융 신용 스코어링 시스템 — 학습부터 실시간 여신 심사 API까지

실제 대출 130만 건(Lending Club)으로 연체를 예측하는 모델을 학습해 **FastAPI로 실시간 서빙**하는
여신 심사 시스템입니다 — 대출 신청을 받아 *연체확률·신용점수·승인 결정·사유코드*를 반환합니다
(`POST /score`). 학습/서빙 분리·Docker 배포·SHAP 설명가능성을 갖췄습니다. 더해, 합성 카드거래
1,330만 건으로 *소비 행동만으로 소득을 추정*하며 데이터 적정성·일반화를 검증합니다.
관통하는 정체성: **"좋아 보이는 수치를 의심하고 정직하게 평가한다."**
📄 **30초 요약**: [`reports/SUMMARY.md`](reports/SUMMARY.md) · 상세: [`reports/project_report.md`](reports/project_report.md).

![핵심 발견 대시보드](reports/figures/00_dashboard.png)

> 6개 핵심 발견: ① 누수의 대가(0.999 가짜→0.701 정직) ② 고도화 이득 +0.004(천장은 데이터)
> ③ 해석모델이 트리와 격차 0.01 ④ 소득은 '얼마'보다 '어떻게'(+230%) ⑤ thin-file 역설
> ⑥ 분할만 바꿔도 PR-AUC 2.3배 착시. 생성: `python src/visualize.py`

## 🎯 한 줄 요약

실제 대출 연체 예측 모델을 학습→**API 서빙**하는 여신 심사 시스템(AUC 0.70, 누수 제거·SHAP 사유코드)
+ 합성 카드거래로 소비 행동만으로 연소득을 **R² 0.56**로 추정. 추천·사기는 *데이터 한계를 증명하는
판단의 챕터*로 배치. 일관된 정체성: 시간분할·베이스라인 대비·신뢰구간으로 **정직하게 평가**.

## 🧭 이 프로젝트의 차별점

- **데이터가 문제를 골랐다.** 신용점수·부채는 행동으로 학습 불가(R²≤0)였고 *소득*만 학습됨을
  검증한 뒤 타깃 선택. "타깃을 증명 없이 고르지 않는다."
- **소득은 '얼마'가 아니라 '어떻게' 쓰는가에 있다.** 총지출만 쓰면 R² 0.17, 행동 풍부함까지
  넣으면 0.56 (**+230%**).
- **가장 필요한 고객에게 가장 안 된다(역설).** thin-file(신규) 고객은 이력이 짧아 추정 거의 불가.
- **"좋아 보이는 수치"를 의심한다.** 추천은 인기도가 행렬분해를 이기고, 사기는 랜덤 분할이
  성능을 2.3배 부풀림을 정량 증명.

## 🏦 본편 — 거래 행동 기반 소득 추정 (정직한 5-fold CV, 95% CI)

| 모델 | CV R² | 중앙 오차 | ±20% 적중 |
|---|---|---|---|
| 평균 예측(naive) | −0.00 | $9,478 | 41.7% |
| 총지출만 | 0.170 | $9,329 | 43.9% |
| **전체 행동(26피처)** | **0.562 [0.46, 0.62]** | **$7,076** | **54.7%** |

- 💡 소득 신호 1위는 총지출이 아니라 **지출 변동성·지리적 이동(주 수)·가맹점 다양성**.
- 🎯 고소득(상위25%) 식별 **AUC 0.892** (신용한도 배정 등 의사결정용).
- 🧪 thin-file 역설: 첫 100건 R² 0.08 → 전체 0.56. 신규 고객엔 추가 신호 필요.
- ⚖️ 연령 편향: 청년 소득 과소(−$3K)·노년 과대(+$4.3K) → adverse-action(규제) 점검.
- 🔬 **일반화 검증**(`income_validation.py`): 분포 이동 스트레스 테스트 → *지역·피처엔 견고,
  소비분포 이동엔 붕괴(covariate shift)*. 합성→실데이터 이전 시 재캘리브레이션 필요함을 정량화.
- 실행: `python src/income_estimation.py` · `python src/income_validation.py`

## 🏦 실데이터 본편 — Lending Club 신용위험(연체) 예측 (`src/credit_risk.py`)

합성 데이터의 약점을 보완하기 위해 **실제 P2P 대출 130만 건**(2007–2018, 연체율 20%)으로
연체를 예측. 시간 분할(과거→미래).

| 모델 | ROC-AUC | 의미 |
|---|---|---|
| 누수 피처 포함 | **0.9998** | 가짜('커닝' — 대출 종료 후 정보 사용) |
| **신청시점 차주정보만** | **0.701** | **정직한 실력** (실제 LC 모델 수준) |
| + LC 자체 등급/금리 | 0.704 | LC 독자 신용등급이 거의 가치 없음(흥미로운 발견) |

- 💡 **누수의 대가 = AUC 0.30**(0.999→0.701)을 실데이터로 실증. 우리의 "누수 탐지·정직 평가"
  스킬이 실데이터에서도 작동함을 증명.
- 💰 승인율 80% 운영 시 승인군 연체율 21.7%→**17.2%**(0.79배). 실제 여신 승인 정책으로 직결.
- ⚖️ 저소득군 연체율 25.5% vs 고소득 18.3% → 금융포용·adverse-action 트레이드오프 명시.
- 🔧 **모델 고도화**(`credit_risk_advanced.py`): 피처 엔지니어링·네이티브 범주형·튜닝의 ablation
  결과 AUC +0.004(미미) → *성능 천장은 알고리즘이 아닌 데이터*임을 실증. 진짜 수확은 **해석가능
  로지스틱이 트리와 격차 0.01**(규제용 설명모델로 충분), **롤링검증 0.716±0.010**(시간 견고),
  *캘리브레이션은 오히려 악화*(무지성 보정의 위험)까지 객관적으로 측정.
- 🏦 **신용 스코어카드**(`credit_scorecard.py`): WOE/IV+PDO 점수표(은행 실무 형태). 점수 구간별
  연체율 **34.8%→10.9% 단조 감소**(좋은 카드). IV 1위는 직접 만든 `loan_to_income`(0.12).
- 🔍 **SHAP 설명가능성**(`shap_explain.py`): 개별 대출 사유코드(*"60개월 장기·높은 상환부담→위험
  75%"*) → ECOA adverse-action 고지에 직결. ([차트](reports/figures/07_shap_summary.png))
- 데이터: `kaggle datasets download -d adarshsng/lending-club-loan-data-csv -p data_lending --unzip`

## 📊 데이터셋 (2026-06 전수 스캔 검증)

- **거래**: **13,305,915건** (2010-01-01 ~ 2019-10-31, 약 10년)
- **사용자**: 1,219명 (소득·신용점수·부채 포함)
- **카드**: 6,146개 (한도·다크웹 노출 여부 포함)
- **가맹점**: 74,831개 / MCC 109개 카테고리
- **사기 라벨**: 8,914,963건, **사기율 0.1495%** (극단적 불균형)

> ⚠️ 과거 문서의 "150만 건 / 2010–2012 / 사용자당 23회"는 오류였으며 위 수치로 정정함.

## 📎 챕터 B — 이상거래 탐지: "데이터가 지탱하지 못함을 증명" (정직한 시간 분할)

| 모델 | ROC-AUC | PR-AUC (무작위 대비) |
|------|---------|--------|
| LogisticRegression (베이스라인) | 0.68 | 0.0043 |
| **LightGBM (메인)** | **0.72** | **0.019 (11배)** |

| 검토 예산 | 사기 적발률 | 정밀도 | 무작위 대비 |
|---|---|---|---|
| **상위 0.5%** | **14.5%** | 5.1% | 29배 |
| 상위 1.0% | 22.3% | 3.9% | 22배 |

- 💰 **비즈니스 환산**: 사기 피해 $291K 중 **$56K 차단(손실 절감 19.3%)**, 전체 거래의 0.5%만 검토.
- 🔧 **velocity 피처**(카드별 경과시간·신규가맹점·지역변경)로 PR-AUC를 **3배(0.006→0.019)** 향상.
- ⚖️ **평가 정직성**: 같은 데이터를 *랜덤 분할*로 보면 PR-AUC 0.044/ROC 0.90으로 **2.3배 부풀려짐**
  (`python src/fraud_detection.py --split random`로 재현). 시간 분할이 실제 운영 성능.

> 불균형 데이터에서는 ROC-AUC가 과대평가되므로 **PR-AUC를 주 지표**로 채택.

## 📎 챕터 A — 추천: "개인화가 항상 답은 아니다" (전체 1,219명, 95% 신뢰구간)

| 모델 | HR@10 | NDCG@10 | 인기도 대비 |
|---|---|---|---|
| Random | 0.006 | 0.001 | −99.9% |
| **인기도(baseline)** | **0.999** | **0.768** | 0% |
| SVD | 0.999 | 0.687 | −10.5% |
| NMF | 0.999 | 0.726 | −5.4% |

> 💡 **단순 인기도가 행렬분해를 능가**한다. 주유소·마트 등 보편 가맹점이 소비를 지배해
> 가맹점 단위 개인화의 한계 효용이 낮기 때문 → 추천 가치는 카테고리/타이밍 단위 또는 사기탐지로.

## 🚀 서빙 시스템 — 실시간 여신 심사 API (`src/api.py`)

분석에 그치지 않고 **돌아가는 시스템**으로 만들었다. 학습/서빙을 분리(오프라인 학습 →
가벼운 모델 아티팩트 → FastAPI 서빙)하고, 대출 신청 JSON을 받아 **연체확률·신용점수·의사결정·
사유코드**를 반환한다.

```bash
python src/train_model.py                          # ① 모델 학습 → models/ 생성
uvicorn src.api:app --host 0.0.0.0 --port 8000     # ② 서빙  → http://localhost:8000/docs
# 또는 컨테이너로:  docker build -t credit-scoring-api .  &&  docker run -p 8000:8000 credit-scoring-api
```

요청/응답 예시:
```bash
curl -X POST localhost:8000/score -H "Content-Type: application/json" -d \
 '{"loan_amnt":35000,"annual_inc":32000,"dti":28,"term":60,"emp_length":1,
   "home_ownership":"RENT","purpose":"small_business","installment":820,
   "open_acc":6,"revol_util":78,"total_acc":9}'
```
```json
{"default_probability":0.527, "credit_score":510, "decision":"거절",
 "reason_codes":[{"factor":"대출 기간(개월)","value":60,"effect":"위험↑"},
                 {"factor":"대출 목적","value":"small_business","effect":"위험↑"}, ...]}
```

- **학습/서빙 분리**: 컨테이너엔 1.2GB 데이터가 아닌 1.7MB 모델만 → 가볍게 배포(Cloud).
- **사유코드(SHAP)**: 개별 결정 근거 반환 → ECOA adverse-action 대응.
- **의사결정 3분기**(승인/검토/거절) + Pydantic 입력 검증 + Swagger UI(`/docs`).

## 🏗️ 구조

```
finance/
├── README.md
├── requirements.txt            # 개발 전체 의존성
├── requirements-api.txt        # 서빙 전용 최소 의존성(컨테이너용)
├── Dockerfile / .dockerignore  # 컨테이너 배포
├── data/                       # 합성 데이터(transactions_data.csv 등, 대용량 .gitignore)
├── data_lending/               # 실데이터(Lending Club loan.csv, .gitignore — 별도 다운로드)
├── models/                     # 학습된 모델 아티팩트(credit_model.txt, model_meta.json)
├── src/
│   ├── api.py                       # ⭐ FastAPI 실시간 여신 심사 API(/score, /health, /docs)
│   ├── train_model.py               # ⭐ 서빙용 모델 학습·저장 → models/
│   ├── credit_risk.py               # ⭐⭐ 실데이터 본편: Lending Club 연체 예측(누수 실증+정직 AUC)
│   ├── credit_risk_advanced.py      # 모델 고도화: FE·튜닝·캘리브레이션·롤링검증·로지스틱비교(ablation)
│   ├── credit_scorecard.py          # 신용 스코어카드(WOE/IV+PDO 점수표) — 은행 실무 형태
│   ├── shap_explain.py              # SHAP 설명가능성(사유코드) — adverse-action용
│   ├── income_estimation.py         # ⭐⭐ 합성 본편: 거래행동→소득 추정(타깃검증+CI+thin-file+공정성)
│   ├── income_validation.py         # 소득모델 일반화 스트레스 테스트(분포 이동)
│   ├── recommendation_eval.py       # 챕터: 추천 정직한 재평가(인기도가 MF를 이김)
│   ├── fraud_detection.py           # 챕터: 이상거래 탐지(temporal split, 데이터 한계 증명)
│   ├── visualize.py                 # 핵심 발견 차트 생성 → reports/figures/*.png
│   └── (legacy) day1~4_*.py, main.py, weekend_visualization.py  # 옛 코드(새 모듈로 대체됨)
└── reports/
    ├── SUMMARY.md                  # 30초 요약(여정·두 기둥·핵심 수치)
    ├── project_report.md           # 상세 보고서(전 섹션)
    ├── figures/                    # 00_dashboard·05_journey·06_scorecard·07_shap 등 차트 8장
    └── *.json                      # 모델별 결과(credit_risk·income·recommendation·fraud ...)
```

## 🔧 사용법

```bash
pip install -r requirements.txt
# (macOS, lightgbm용 OpenMP) brew install libomp

# 실데이터 신용위험 (먼저 Lending Club 다운로드)
kaggle datasets download -d adarshsng/lending-club-loan-data-csv -p data_lending --unzip
python src/credit_risk.py            # v1: 누수 실증 + 정직 AUC
python src/credit_risk_advanced.py   # 고도화: ablation·튜닝·캘리브레이션·롤링·로지스틱비교
python src/credit_scorecard.py       # 신용 스코어카드(WOE/IV+PDO 점수표)
python src/shap_explain.py           # SHAP 설명가능성(사유코드)

# 합성 데이터 — 소득 추정 + 일반화 검증
python src/income_estimation.py
python src/income_validation.py

# 챕터 — 추천 / 사기
python src/recommendation_eval.py
python src/fraud_detection.py                  # 정직한 시간 분할
python src/fraud_detection.py --split random   # 랜덤 분할(낙관 편향) 비교

# 핵심 발견 시각화 (결과 JSON만 읽어 차트 생성, 모델 재실행 불필요)
python src/visualize.py

# 실시간 여신 심사 API (학습 → 서빙)
python src/train_model.py                        # 모델 학습·저장 → models/
uvicorn src.api:app --host 0.0.0.0 --port 8000   # → http://localhost:8000/docs
docker build -t credit-scoring-api . && docker run -p 8000:8000 credit-scoring-api  # 컨테이너 배포
```

## ⚠️ 정직한 한계

- **데이터**: 카드거래는 합성(IBM 생성), Lending Club은 공개·과사용 데이터 → 절대 수치보다
  *방법론·관계·문제 구조*를 주장한다(데이터 현실성과 무관하게 참).
- **성능 천장**: 신용위험 AUC 0.70은 신청정보 기반의 전형적 수준이며 고도화로도 +0.004(천장은
  데이터). 소득 R² 0.56은 합성이라 일부 생성기 산물.
- 사기 탐지 절대 성능은 낮다(상위 0.5%에서 14.5% 적발) → 지리속도·세션·24h 롤링 + 재학습 필요.
- 추천은 implicit feedback이라 "미구매 = 비선호"가 아니다 → ALS/BPR·카테고리 추천으로 확장 여지.

## 🎓 핵심 키워드

**시스템**: FastAPI 서빙 · 학습/서빙 분리 · Docker · 모델 아티팩트 · reason code(adverse-action)
**모델링**: 데이터 누수 차단 · temporal/rolling 검증 · PR-AUC · 도메인 피처 엔지니어링 · 불균형 데이터 ·
신용 스코어카드(WOE/IV) · SHAP 설명가능성 · 확률 캘리브레이션
**평가 정직성**: 베이스라인 대비 lift · 부트스트랩 신뢰구간 · 분포이동(covariate shift) · 비용민감 운영점
