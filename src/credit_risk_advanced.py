#!/usr/bin/env python3
"""
실데이터 신용위험 모델 고도화 (Credit Risk — Advanced)
=====================================================

v1(`credit_risk.py`, 신청시점 차주정보만 AUC 0.701)을 실무급으로 마감한다.
각 개선이 성능에 얼마나 기여하는지 ablation 으로 *객관적으로* 측정한다.

적용한 6가지 개선
  1. 하이퍼파라미터 튜닝  : 시간순 검증셋에서 random search + early stopping
  2. 피처 엔지니어링      : 소득대비대출·상환부담률 등 도메인 비율 피처
  3. 확률 캘리브레이션    : isotonic 보정 → Brier/ECE 로 신뢰도 개선 측정
  4. 범주형 처리          : 임의 정수코드 → LightGBM 네이티브 categorical
  5. 견고한 검증          : 롤링-오리진 시간검증(여러 시점) + 분산
  6. 해석가능 모델 비교   : 로지스틱(스코어카드형) vs LightGBM, 회귀계수=사유코드

모든 평가는 시간 분할(과거 학습→미래 예측). 누수 피처는 일절 사용하지 않는다.

Usage: python src/credit_risk_advanced.py
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data_lending" / "loan.csv"
REPORTS = ROOT / "reports"
SEED = 42

NUM_V1 = ["loan_amnt", "annual_inc", "dti", "open_acc", "revol_util",
          "total_acc", "installment", "term_months", "emp_years"]
ENG = ["loan_to_income", "pay_to_income", "inst_to_loan", "open_to_total",
       "income_per_acc", "log_income", "log_loan"]
CAT = ["home_ownership", "verification_status", "purpose", "application_type", "addr_state"]
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
             "addr_state"])
    print("Lending Club 로딩...")
    df = pd.read_csv(DATA, usecols=cols, low_memory=False)
    df = df[df["loan_status"].isin(BAD | GOOD)].copy()
    df["default"] = df["loan_status"].isin(BAD).astype(int)

    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df["term_months"] = _pct(df["term"].astype(str).str.replace("months", "", regex=False))
    df["revol_util"] = _pct(df["revol_util"])
    emp_map = {"< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
               "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
               "10+ years": 10}
    df["emp_years"] = df["emp_length"].map(emp_map)

    # (2) 피처 엔지니어링: 도메인 비율
    inc = df["annual_inc"].clip(lower=1)
    df["loan_to_income"] = df["loan_amnt"] / inc
    df["pay_to_income"] = (df["installment"] * 12) / inc            # 상환부담률
    df["inst_to_loan"] = df["installment"] / (df["loan_amnt"] + 1)
    df["open_to_total"] = df["open_acc"] / (df["total_acc"] + 1)
    df["income_per_acc"] = df["annual_inc"] / (df["total_acc"] + 1)
    df["log_income"] = np.log1p(df["annual_inc"])
    df["log_loan"] = np.log1p(df["loan_amnt"])

    df = df.dropna(subset=["issue_d"]).sort_values("issue_d").reset_index(drop=True)
    # 범주형: 코드판(정수) + 네이티브판(category)
    for c in CAT:
        df[c] = df[c].fillna("NA").astype(str)
        df[c + "_code"] = df[c].astype("category").cat.codes
        df[c] = df[c].astype("category")
    print(f"  종료 대출 {len(df):,}건 / 연체율 {df['default'].mean()*100:.1f}% / "
          f"기간 {df['issue_d'].min().date()}~{df['issue_d'].max().date()}")
    return df


def auc_pr(y, p):
    from sklearn.metrics import average_precision_score, roc_auc_score
    return float(roc_auc_score(y, p)), float(average_precision_score(y, p))


def fit_lgb(Xtr, ytr, Xva, yva, params, cat_features=None):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(**params, random_state=SEED, n_jobs=-1, verbose=-1)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
          categorical_feature=cat_features or "auto",
          callbacks=[lgb.early_stopping(40, verbose=False)])
    return m


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum():
            e += abs(p[m].mean() - y[m].mean()) * m.sum() / len(y)
    return float(e)


def main():
    print("=" * 66)
    print("실데이터 신용위험 모델 고도화 (v1 AUC 0.701 → ?)")
    print("=" * 66)
    df = load()
    n = len(df)
    i_tr, i_va = int(n * 0.70), int(n * 0.80)
    y = df["default"]
    ytr, yva, yte = y[:i_tr], y[i_tr:i_va], y[i_va:]
    print(f"시간 3분할 | train {i_tr:,} / valid {i_va-i_tr:,} / test {n-i_va:,}")
    DEFAULT = dict(n_estimators=1000, learning_rate=0.05, num_leaves=63)
    out = {"n_loans": int(n), "default_rate_pct": round(float(y.mean()*100), 2),
           "v1_reference_auc": 0.701}

    # ===================== ablation: 개선별 기여 측정 =====================
    print("\n[ablation] 각 개선이 AUC를 얼마나 올리나 (test 평가)")
    def code_cols(cols_cat): return [c + "_code" for c in cols_cat]

    def Xset(num, cat, native):
        catc = cat if native else code_cols(cat)
        return df[num + catc]

    ladder = {}
    # M0: v1 재현 (v1 numeric + 코드 범주형, 기본 파라미터)
    X = Xset(NUM_V1, CAT, native=False)
    m = fit_lgb(X[:i_tr], ytr, X[i_tr:i_va], yva, DEFAULT)
    ladder["M0_v1재현"] = auc_pr(yte, m.predict_proba(X[i_va:])[:, 1])
    # M1: + 피처 엔지니어링
    X = Xset(NUM_V1 + ENG, CAT, native=False)
    m = fit_lgb(X[:i_tr], ytr, X[i_tr:i_va], yva, DEFAULT)
    ladder["M1_+피처엔지니어링"] = auc_pr(yte, m.predict_proba(X[i_va:])[:, 1])
    # M2: + 네이티브 범주형
    X = Xset(NUM_V1 + ENG, CAT, native=True)
    m2 = fit_lgb(X[:i_tr], ytr, X[i_tr:i_va], yva, DEFAULT, cat_features=CAT)
    ladder["M2_+네이티브범주형"] = auc_pr(yte, m2.predict_proba(X[i_va:])[:, 1])

    # ===================== (1) 하이퍼파라미터 튜닝 =====================
    print("[튜닝] valid 에서 random search (subsample) ...")
    rng = np.random.default_rng(SEED)
    grid = dict(num_leaves=[31, 63, 127, 255], learning_rate=[0.02, 0.03, 0.05, 0.08],
                min_child_samples=[20, 50, 100, 200], subsample=[0.7, 0.8, 0.9],
                colsample_bytree=[0.7, 0.8, 0.9], reg_lambda=[0.0, 1.0, 5.0])
    Xfull = Xset(NUM_V1 + ENG, CAT, native=True)
    # 탐색은 학습구간 일부로 빠르게
    sub = rng.choice(i_tr, size=min(i_tr, 300000), replace=False)
    Xs, ys = Xfull.iloc[sub], y.iloc[sub]
    best, best_auc, best_params = None, -1, None
    for _ in range(12):
        p = {k: rng.choice(v).item() for k, v in grid.items()}
        p["n_estimators"] = 1500
        mm = fit_lgb(Xs, ys, Xfull[i_tr:i_va], yva, p, cat_features=CAT)
        a = auc_pr(yva, mm.predict_proba(Xfull[i_tr:i_va])[:, 1])[0]
        if a > best_auc:
            best_auc, best_params = a, p
    print(f"  best valid AUC {best_auc:.4f} | params {best_params}")

    # M3: 튜닝된 최종 모델 (train 전체로 재학습, test 평가)
    final = fit_lgb(Xfull[:i_tr], ytr, Xfull[i_tr:i_va], yva, best_params, cat_features=CAT)
    p_te = final.predict_proba(Xfull[i_va:])[:, 1]
    ladder["M3_+튜닝(최종)"] = auc_pr(yte, p_te)

    out["ablation"] = {k: {"roc_auc": round(v[0], 4), "pr_auc": round(v[1], 4)}
                       for k, v in ladder.items()}
    prev = None
    for k, (a, pr) in ladder.items():
        delta = f"({(a-prev)*+1:+.4f})" if prev is not None else ""
        print(f"  {k:22s} AUC {a:.4f}  PR {pr:.4f}  {delta}")
        prev = a
    out["best_params"] = {k: (v.item() if hasattr(v, "item") else v) for k, v in best_params.items()}

    # ===================== (3) 확률 캘리브레이션 =====================
    print("\n[캘리브레이션] valid 로 isotonic 보정 → test 신뢰도")
    from sklearn.isotonic import IsotonicRegression
    p_va = final.predict_proba(Xfull[i_tr:i_va])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_va, yva)
    p_te_cal = iso.predict(p_te)
    brier_raw = float(np.mean((p_te - yte.to_numpy()) ** 2))
    brier_cal = float(np.mean((p_te_cal - yte.to_numpy()) ** 2))
    ece_raw, ece_cal = ece(yte.to_numpy(), p_te), ece(yte.to_numpy(), p_te_cal)
    out["calibration"] = {"brier_raw": round(brier_raw, 5), "brier_cal": round(brier_cal, 5),
                          "ece_raw": round(ece_raw, 4), "ece_cal": round(ece_cal, 4)}
    print(f"  Brier {brier_raw:.5f} → {brier_cal:.5f} | ECE {ece_raw:.4f} → {ece_cal:.4f} "
          f"(낮을수록 좋음)")

    # ===================== (6) 해석가능 모델 비교 =====================
    print("\n[해석가능] 로지스틱(스코어카드형) vs LightGBM")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    num = NUM_V1 + ENG
    # 로지스틱: 수치 표준화 + 범주 원핫
    ohe = pd.get_dummies(df[CAT], drop_first=True).astype("float32")
    Xlr = pd.concat([df[num].apply(pd.to_numeric, errors="coerce").astype("float32"),
                     ohe], axis=1).fillna(0)
    sub_tr = rng.choice(i_tr, size=min(i_tr, 300000), replace=False)
    sc = StandardScaler()
    lr = LogisticRegression(max_iter=2000, C=1.0)
    lr.fit(sc.fit_transform(Xlr.iloc[sub_tr]), y.iloc[sub_tr])
    p_lr = lr.predict_proba(sc.transform(Xlr.iloc[i_va:]))[:, 1]
    lr_auc = auc_pr(yte, p_lr)
    out["logistic_vs_lgbm"] = {"logistic_auc": round(lr_auc[0], 4),
                               "lgbm_auc": round(ladder["M3_+튜닝(최종)"][0], 4),
                               "gap": round(ladder["M3_+튜닝(최종)"][0] - lr_auc[0], 4)}
    print(f"  로지스틱 AUC {lr_auc[0]:.4f} | LightGBM AUC {ladder['M3_+튜닝(최종)'][0]:.4f} "
          f"| 격차 {ladder['M3_+튜닝(최종)'][0]-lr_auc[0]:+.4f}")
    # 사유코드: 위험을 키우는/줄이는 계수 상위
    coef = pd.Series(lr.coef_[0], index=Xlr.columns).sort_values()
    risk_up = coef.tail(5)[::-1]; risk_down = coef.head(5)
    out["logistic_reason_codes"] = {"risk_up": {k: round(float(v), 3) for k, v in risk_up.items()},
                                    "risk_down": {k: round(float(v), 3) for k, v in risk_down.items()}}
    print("  연체위험 ↑ 요인:", ", ".join(f"{k}({v:+.2f})" for k, v in risk_up.items()))
    print("  연체위험 ↓ 요인:", ", ".join(f"{k}({v:+.2f})" for k, v in risk_down.items()))

    # ===================== (5) 롤링-오리진 시간검증 =====================
    print("\n[롤링검증] 과거→다음해 예측을 여러 시점으로 (분산 확인)")
    df["yr"] = df["issue_d"].dt.year
    roll = {}
    for test_yr in [2014, 2015, 2016]:
        tr_mask = df["yr"] < test_yr
        te_mask = df["yr"] == test_yr
        if te_mask.sum() < 5000 or tr_mask.sum() < 20000:
            continue
        mm = fit_lgb(Xfull[tr_mask.values], y[tr_mask.values],
                     Xfull[te_mask.values], y[te_mask.values], best_params, cat_features=CAT)
        a = auc_pr(y[te_mask.values], mm.predict_proba(Xfull[te_mask.values])[:, 1])[0]
        roll[str(test_yr)] = round(a, 4)
        print(f"  ~{test_yr-1} 학습 → {test_yr} 예측: AUC {a:.4f} (n_test={int(te_mask.sum()):,})")
    vals = list(roll.values())
    out["rolling_validation"] = {"per_year": roll,
                                 "mean": round(float(np.mean(vals)), 4),
                                 "std": round(float(np.std(vals)), 4)}
    print(f"  롤링 평균 AUC {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # ===================== 종합 =====================
    m0 = ladder["M0_v1재현"][0]; m3 = ladder["M3_+튜닝(최종)"][0]
    out["summary"] = {"v1_auc": round(m0, 4), "final_auc": round(m3, 4),
                      "improvement": round(m3 - m0, 4)}
    print("\n" + "=" * 66)
    print(f"종합: v1 재현 AUC {m0:.4f} → 고도화 최종 {m3:.4f}  ({m3-m0:+.4f})")
    print(f"  + 확률 보정(ECE {ece_raw:.3f}→{ece_cal:.3f}), 롤링검증 {np.mean(vals):.3f}±{np.std(vals):.3f},")
    print(f"  + 해석가능 로지스틱 {lr_auc[0]:.3f}(트리와 격차 {m3-lr_auc[0]:+.3f})")
    print("=" * 66)

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "credit_risk_advanced_results.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    print("결과 저장: reports/credit_risk_advanced_results.json")


if __name__ == "__main__":
    main()
