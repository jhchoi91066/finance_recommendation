#!/usr/bin/env python3
"""
서빙용 신용위험 모델 학습·저장 (오프라인 1회)
=============================================

Lending Club 실데이터로 연체 예측 모델을 학습하고, 서빙(api.py)이 로드할 가벼운
아티팩트를 `models/` 에 저장한다. 신청 시점 차주 정보만 사용(누수 차단).

산출
  models/credit_model.txt   LightGBM 모델(네이티브)
  models/model_meta.json    피처 순서·범주 인코딩·점수 파라미터·임계값

Usage: python src/train_model.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data_lending" / "loan.csv"
MODELS = ROOT / "models"
SEED = 42

# 모델 입력 피처 순서 (서빙도 이 순서를 그대로 사용)
FEATURE_ORDER = ["loan_amnt", "annual_inc", "dti", "open_acc", "revol_util", "total_acc",
                 "installment", "term_months", "emp_years", "loan_to_income",
                 "pay_to_income", "inst_to_loan", "home_ownership_code", "purpose_code"]
CATS = ["home_ownership", "purpose"]
BAD = {"Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off"}
GOOD = {"Fully Paid", "Does not meet the credit policy. Status:Fully Paid"}
# 점수 스케일(스코어카드와 동일) / 의사결정 임계값
PDO, BASE, BASE_ODDS = 20, 600, 20
THRESH = {"approve_below": 0.20, "reject_above": 0.35}   # p<0.20 승인, p>0.35 거절, 사이 검토


def _pct(s):
    return pd.to_numeric(s.astype(str).str.replace("%", "", regex=False), errors="coerce")


def build(df):
    df = df.copy()
    df["term_months"] = _pct(df["term"].astype(str).str.replace("months", "", regex=False))
    df["revol_util"] = _pct(df["revol_util"])
    emp = {"< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4, "5 years": 5,
           "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9, "10+ years": 10}
    df["emp_years"] = df["emp_length"].map(emp)
    inc = df["annual_inc"].clip(lower=1)
    df["loan_to_income"] = df["loan_amnt"] / inc
    df["pay_to_income"] = (df["installment"] * 12) / inc
    df["inst_to_loan"] = df["installment"] / (df["loan_amnt"] + 1)
    return df


def main():
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score

    print("학습 데이터 로딩...")
    cols = ["loan_status", "issue_d", "loan_amnt", "annual_inc", "dti", "open_acc",
            "revol_util", "total_acc", "term", "emp_length", "installment",
            "home_ownership", "purpose"]
    df = pd.read_csv(DATA, usecols=cols, low_memory=False)
    df = df[df["loan_status"].isin(BAD | GOOD)].copy()
    df["default"] = df["loan_status"].isin(BAD).astype(int)
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df = df.dropna(subset=["issue_d"]).sort_values("issue_d").reset_index(drop=True)
    df = build(df)

    # 범주 인코딩 맵 저장 (서빙에서 동일 매핑)
    cat_maps = {}
    for c in CATS:
        cats = list(pd.Categorical(df[c].astype(str)).categories)
        cat_maps[c] = cats
        code = {v: i for i, v in enumerate(cats)}
        df[c + "_code"] = df[c].astype(str).map(code).fillna(-1).astype(int)

    X = df[FEATURE_ORDER].apply(pd.to_numeric, errors="coerce").fillna(0)
    y = df["default"]
    # 시간 분할로 검증 AUC 측정(정직성), 최종 모델은 전체로 학습
    split = int(len(df) * 0.8)
    val_model = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                               min_child_samples=50, subsample=0.9, colsample_bytree=0.7,
                               reg_lambda=5.0, random_state=SEED, n_jobs=-1, verbose=-1)
    val_model.fit(X.iloc[:split], y.iloc[:split])
    auc = roc_auc_score(y.iloc[split:], val_model.predict_proba(X.iloc[split:])[:, 1])
    print(f"검증 AUC(시간분할) = {auc:.4f}")

    print("최종 모델 학습(전체 데이터)...")
    model = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,
                           min_child_samples=50, subsample=0.9, colsample_bytree=0.7,
                           reg_lambda=5.0, random_state=SEED, n_jobs=-1, verbose=-1)
    model.fit(X, y)

    MODELS.mkdir(exist_ok=True)
    model.booster_.save_model(str(MODELS / "credit_model.txt"))
    meta = {
        "feature_order": FEATURE_ORDER,
        "cat_maps": cat_maps,
        "score": {"pdo": PDO, "base": BASE, "base_odds": BASE_ODDS},
        "thresholds": THRESH,
        "default_rate": round(float(y.mean()), 4),
        "validation_auc": round(float(auc), 4),
        "n_train": int(len(df)),
        "feature_labels": {  # 사유코드용 한글 라벨
            "loan_amnt": "대출 금액", "annual_inc": "연소득", "dti": "부채/소득비(DTI)",
            "open_acc": "활성 계좌 수", "revol_util": "신용 회전율", "total_acc": "총 계좌 수",
            "installment": "월 상환액", "term_months": "대출 기간(개월)",
            "emp_years": "근속연수", "loan_to_income": "소득대비 대출액",
            "pay_to_income": "상환부담률", "inst_to_loan": "상환액/대출액 비율",
            "home_ownership_code": "주거 형태", "purpose_code": "대출 목적",
        },
    }
    (MODELS / "model_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"저장: {MODELS}/credit_model.txt, model_meta.json")
    print("✅ 모델 학습·저장 완료")


if __name__ == "__main__":
    main()
