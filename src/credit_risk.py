#!/usr/bin/env python3
"""
실데이터 신용위험(연체) 예측 — Lending Club (Real-data Credit Default)
======================================================================

합성 데이터의 한계를 보완하기 위해, **실제 P2P 대출 226만 건**(Lending Club, 2007–2018)
으로 연체(default)를 예측한다. 우리가 합성 데이터에서 갈고닦은 *정직한 평가* 원칙을
**실데이터에서 그대로 재현**하는 것이 목적이다.

이 모듈의 핵심 = 데이터 누수(leakage)를 실증하고 제거한다
  Lending Club 데이터엔 '대출이 끝난 뒤에야 알 수 있는' 컬럼(회수금·총상환액·미상환원금
  등)이 섞여 있다. 이를 피처로 쓰면 AUC가 0.99로 *가짜로* 치솟는다(미래를 보고 과거를
  맞히는 셈). 신청 시점에 실제로 알 수 있는 정보만 써야 정직한 성능이 나온다.
    모델 L  : 누수 피처 포함        → AUC ~0.99 (가짜, '커닝')
    모델 A  : 신청시점 차주 정보만   → 정직한 AUC
    모델 A+ : A + LC 자체 신용등급   → LC의 리스크 가격책정이 더하는 가치

평가 원칙(다른 모듈과 일관)
  - 시간 분할: 과거 대출로 학습 → 미래 대출의 연체 예측(현실적, 누수 차단).
  - 베이스라인 대비 / PR-AUC 병기(연체율 ~20%) / 승인율 기반 운영점 / 세그먼트 공정성.

Usage: python src/credit_risk.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data_lending" / "loan.csv"
REPORTS = ROOT / "reports"
SEED = 42

# 신청 시점에 알 수 있는 차주 정보 (LC 등급/금리 제외)
APP_NUM = ["loan_amnt", "annual_inc", "dti", "open_acc", "revol_util",
           "total_acc", "installment", "term_months", "emp_years"]
APP_CAT = ["home_ownership", "verification_status", "purpose",
           "application_type", "addr_state"]
LC_PRICING = ["int_rate", "grade_code", "subgrade_code"]      # LC 자체 리스크 가격
LEAKY = ["recoveries", "collection_recovery_fee", "total_pymnt", "total_rec_prncp",
         "total_rec_int", "out_prncp", "last_pymnt_amnt", "total_pymnt_inv"]

BAD = {"Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off"}
GOOD = {"Fully Paid", "Does not meet the credit policy. Status:Fully Paid"}


def _pct(s):
    return pd.to_numeric(s.astype(str).str.replace("%", "", regex=False), errors="coerce")


def load():
    cols = (["loan_status", "issue_d", "loan_amnt", "annual_inc", "dti", "open_acc",
             "revol_util", "total_acc", "installment", "term", "emp_length",
             "home_ownership", "verification_status", "purpose", "application_type",
             "addr_state", "int_rate", "grade", "sub_grade"] + LEAKY)
    print("Lending Club 로딩...")
    df = pd.read_csv(DATA, usecols=cols, low_memory=False)
    print(f"  전체 {len(df):,}건")

    # --- 타깃: 종료된 대출만 (Current/Late 등 미결 제외) ---
    df = df[df["loan_status"].isin(BAD | GOOD)].copy()
    df["default"] = df["loan_status"].isin(BAD).astype(int)
    print(f"  종료 대출 {len(df):,}건 / 연체율 {df['default'].mean()*100:.1f}%")

    # --- 파싱 ---
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df["term_months"] = _pct(df["term"].astype(str).str.replace("months", "", regex=False))
    df["int_rate"] = _pct(df["int_rate"])
    df["revol_util"] = _pct(df["revol_util"])
    emp_map = {"< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
               "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
               "10+ years": 10}
    df["emp_years"] = df["emp_length"].map(emp_map)
    df["grade_code"] = df["grade"].astype("category").cat.codes
    df["subgrade_code"] = df["sub_grade"].astype("category").cat.codes
    for c in APP_CAT:
        df[c] = df[c].astype("category").cat.codes
    df = df.dropna(subset=["issue_d"]).sort_values("issue_d").reset_index(drop=True)
    return df


def evaluate(df, feature_cols, name, split_idx):
    from lightgbm import LGBMClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    y = df["default"]
    Xtr, Xte = X.iloc[:split_idx], X.iloc[split_idx:]
    ytr, yte = y.iloc[:split_idx], y.iloc[split_idx:]
    m = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63,
                       subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                       n_jobs=-1, verbose=-1)
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p)
    pr = average_precision_score(yte, p)
    print(f"  [{name:22s}] ROC-AUC {auc:.4f} | PR-AUC {pr:.4f}  ({len(feature_cols)} 피처)")
    return {"roc_auc": round(float(auc), 4), "pr_auc": round(float(pr), 4),
            "n_features": len(feature_cols)}, m, X, yte, p, split_idx


def main():
    print("=" * 64)
    print("실데이터 신용위험(연체) 예측 — Lending Club")
    print("=" * 64)
    df = load()
    split = int(len(df) * 0.8)
    print(f"시간 분할 | 학습 {df['issue_d'].iloc[0].date()}~{df['issue_d'].iloc[split-1].date()} "
          f"({split:,}) | 검증 ~{df['issue_d'].iloc[-1].date()} ({len(df)-split:,})")

    out = {"n_loans": int(len(df)), "default_rate_pct": round(float(df["default"].mean()*100), 2)}

    print("\n[모델 비교] 누수 vs 정직")
    out["model_leaky"], *_ = evaluate(df, APP_NUM + APP_CAT + LEAKY, "L: 누수 포함(가짜)", split)
    out["model_app_only"], m_app, Xapp, yte, p_app, si = evaluate(df, APP_NUM + APP_CAT, "A: 신청시점 차주정보만", split)
    out["model_app_plus_lc"], *_ = evaluate(df, APP_NUM + APP_CAT + LC_PRICING, "A+: A + LC등급/금리", split)

    leaky_auc = out["model_leaky"]["roc_auc"]; app_auc = out["model_app_only"]["roc_auc"]
    print(f"\n  → 누수의 대가: 누수 포함 AUC {leaky_auc:.3f} 는 '커닝'. 신청시점만 쓰면 {app_auc:.3f} 가"
          f" 정직한 실력. (격차 {leaky_auc-app_auc:+.3f})")

    # --- 베이스라인(연체율 상수) AUC=0.5 명시 ---
    out["baseline_auc"] = 0.5

    # --- 승인율 기반 운영점 + 비즈니스 환산 (정직 모델 A) ---
    print("\n[운영점] 신청시점 모델로 위험순 정렬 → 가장 위험한 X% 거절 시")
    yte_a = yte.to_numpy()
    base_def = yte_a.mean()
    order = np.argsort(p_app)  # 위험 낮은 순
    ops = {}
    for approve in (0.90, 0.80, 0.70):
        k = int(len(p_app) * approve)
        accepted = order[:k]                      # 위험 낮은 상위 approve 만 승인
        acc_def = yte_a[accepted].mean()
        ops[f"approve_{int(approve*100)}pct"] = {
            "accepted_default_rate_pct": round(float(acc_def*100), 2),
            "vs_overall": round(float(acc_def/base_def), 2)}
        print(f"  승인율 {int(approve*100)}% → 승인군 연체율 {acc_def*100:.2f}% "
              f"(전체 {base_def*100:.2f}% 대비 {acc_def/base_def:.2f}배)")
    out["operating_points"] = ops

    # --- 세그먼트 공정성: 소득구간별 실제 연체율 ---
    print("\n[공정성] 소득 구간별 실제 연체율 (모델이 소득에 의존하면 편향 위험)")
    te = df.iloc[si:].copy()
    te["inc_band"] = pd.qcut(te["annual_inc"].clip(upper=te["annual_inc"].quantile(0.99)),
                             4, labels=["저소득", "중하", "중상", "고소득"], duplicates="drop")
    seg = te.groupby("inc_band", observed=True)["default"].agg(["mean", "size"])
    fair = {}
    for band, row in seg.iterrows():
        fair[str(band)] = {"default_rate_pct": round(float(row["mean"]*100), 2), "n": int(row["size"])}
        print(f"  {band}: 연체율 {row['mean']*100:.2f}% (n={int(row['size']):,})")
    out["income_band_default"] = fair

    # --- 피처 중요도(정직 모델) ---
    imp = pd.Series(m_app.feature_importances_, index=(APP_NUM + APP_CAT)).sort_values(ascending=False)
    out["top_features_app"] = {k: int(v) for k, v in imp.head(10).items()}
    print("\n[정직 모델 상위 피처]\n", imp.head(10).to_string())

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "credit_risk_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("\n결과 저장: reports/credit_risk_results.json")
    print("✅ 실데이터 신용위험 파이프라인 완료")


if __name__ == "__main__":
    main()
