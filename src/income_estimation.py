#!/usr/bin/env python3
"""
거래 행동 기반 소득 추정 (Transaction-based Income Estimation)
==============================================================

이 프로젝트의 척추. "고객이 *어떻게 쓰는가*(10년 거래)만으로 그 사람의 *소득*을
추정할 수 있는가?" — 페이스테이트먼트가 없는 thin-file·긱워커 고객의 대안 소득검증/
신용평가에 쓰이는 실제 핀테크 과제다.

왜 소득인가 (데이터가 문제를 골랐다)
  행동 피처로 4개 재무 타깃을 예측한 CV R²:
    credit_score −0.19 / log_debt −0.21 / dti −0.14  → 학습 불가(합성 생성기가 거의 독립 부여)
    log_income  +0.39                                  → 강한 신호
  → 신용점수가 아니라 *소득*이 이 데이터가 정직하게 지탱하는 타깃이다.
  (이 검증 과정 자체가 '데이터 적정성을 판별하는 능력'의 증거다.)

평가 원칙(다른 모듈과 일관)
  - 베이스라인 대비로 말한다: 평균 예측 / '총지출만' 단일피처 / 전체 행동모델.
  - 교차검증 + 부트스트랩 신뢰구간. 단일 수치를 믿지 않는다.
  - thin-file 분석: 거래 이력이 짧을수록 정확도가 어떻게 떨어지는가(실사용 시나리오).
  - 설명가능성(permutation importance) + 세그먼트 공정성(연령·성별 편향).

Usage:
    python src/income_estimation.py
    python src/income_estimation.py --quick   # thin-file 곡선 생략(빠름)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
SEED = 42

# 해석 가능한 카테고리 그룹 (MCC 설명 키워드 매칭)
MCC_GROUPS = {
    "dining": ["restaurant", "eating", "drinking", "fast food", "bar"],
    "travel": ["air", "airline", "hotel", "lodging", "cruise", "travel",
               "car rental", "passenger railway", "tour"],
    "grocery": ["grocery", "supermarket"],
    "fuel": ["service station", "fuel", "gas"],
    "retail_discount": ["discount", "variety", "wholesale"],
    "department": ["department store"],
    "entertainment": ["amusement", "theatrical", "recreation", "motion picture",
                      "sporting", "membership clubs"],
    "health": ["medical", "doctor", "dental", "hospital", "drug", "pharmac"],
    "utilities": ["utilities", "telecommunication", "cable"],
}


def _money(s):
    return (s.astype(str).str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False).replace({"nan": np.nan}).astype(float))


# --------------------------------------------------------------------------- #
# 1. 거래 → 사용자별 행동 피처
# --------------------------------------------------------------------------- #
def load_transactions():
    print("거래 로딩(필요 컬럼만)...")
    tx = pd.read_csv(DATA_DIR / "transactions_data.csv",
                     usecols=["client_id", "date", "amount", "mcc",
                              "use_chip", "merchant_id", "merchant_state"])
    tx["amount"] = _money(tx["amount"]).abs()
    tx["date"] = pd.to_datetime(tx["date"])
    tx["year"] = tx["date"].dt.year
    tx["hour"] = tx["date"].dt.hour
    tx["dow"] = tx["date"].dt.dayofweek
    print(f"  거래 {len(tx):,}건 / 사용자 {tx.client_id.nunique():,}명")

    # MCC → 카테고리 그룹 매핑
    mcc_names = json.loads((DATA_DIR / "mcc_codes.json").read_text())
    code2grp = {}
    for code, name in mcc_names.items():
        low = name.lower()
        for grp, kws in MCC_GROUPS.items():
            if any(k in low for k in kws):
                code2grp[int(code)] = grp
                break
    tx["mcc_grp"] = tx["mcc"].map(code2grp)
    return tx


def build_features(tx: pd.DataFrame) -> pd.DataFrame:
    """사용자별 행동 피처 테이블."""
    g = tx.groupby("client_id")
    f = pd.DataFrame({
        "n_tx": g.size(),
        "total_spend": g["amount"].sum(),
        "mean_amt": g["amount"].mean(),
        "median_amt": g["amount"].median(),
        "std_amt": g["amount"].std(),
        "p90_amt": g["amount"].quantile(0.90),
        "n_merchants": g["merchant_id"].nunique(),
        "n_mcc": g["mcc"].nunique(),
        "n_states": g["merchant_state"].nunique(),
        "n_years": g["year"].nunique(),
        "online_share": g["use_chip"].apply(lambda s: (s == "Online Transaction").mean()),
        "weekend_share": g["dow"].apply(lambda s: (s >= 5).mean()),
        "night_share": g["hour"].apply(lambda s: ((s < 6) | (s >= 22)).mean()),
    })
    f["cv_amt"] = f["std_amt"] / (f["mean_amt"] + 1)
    f["spend_per_year"] = f["total_spend"] / f["n_years"].clip(lower=1)
    f["tx_per_year"] = f["n_tx"] / f["n_years"].clip(lower=1)
    f["merchants_per_year"] = f["n_merchants"] / f["n_years"].clip(lower=1)

    # 카테고리 그룹별 지출 점유율 (해석 가능 피처)
    grp_spend = (tx.dropna(subset=["mcc_grp"])
                 .groupby(["client_id", "mcc_grp"])["amount"].sum().unstack(fill_value=0))
    grp_share = grp_spend.div(f["total_spend"], axis=0).add_prefix("share_")
    f = f.join(grp_share).fillna({c: 0 for c in grp_share.columns})
    return f.reset_index()


# --------------------------------------------------------------------------- #
# 2. 평가 유틸
# --------------------------------------------------------------------------- #
def cv_r2_ci(model_fn, X, y, n_boot=300):
    """5-fold out-of-fold 예측 → R², 그리고 부트스트랩 95% CI + $ 오차."""
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for tr, va in kf.split(X):
        m = model_fn()
        m.fit(X.iloc[tr], y.iloc[tr])
        oof[va] = m.predict(X.iloc[va])
    ss_res = ((y - oof) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    # $ 오차 (로그 타깃을 되돌림)
    yd, pd_ = np.expm1(y.to_numpy()), np.expm1(oof)
    mae = np.median(np.abs(yd - pd_))          # 중앙절대오차($)
    within20 = np.mean(np.abs(pd_ - yd) / yd < 0.20)  # ±20% 이내 비율
    # 부트스트랩 CI for R²
    rng = np.random.default_rng(SEED)
    boots = []
    res = y.to_numpy() - oof
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        sr = (res[idx] ** 2).sum()
        st = ((y.to_numpy()[idx] - y.to_numpy()[idx].mean()) ** 2).sum()
        boots.append(1 - sr / st)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"r2": round(float(r2), 4), "r2_ci95": [round(float(lo), 4), round(float(hi), 4)],
            "median_abs_err_usd": round(float(mae), 0), "within_20pct": round(float(within20), 4)}


# --------------------------------------------------------------------------- #
# 3. 메인
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="thin-file 곡선 생략")
    args = ap.parse_args()

    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.inspection import permutation_importance

    print("=" * 64)
    print("거래 행동 기반 소득 추정")
    print("=" * 64)
    tx = load_transactions()
    feats = build_features(tx)

    users = pd.read_csv(DATA_DIR / "users_data.csv")
    users["yearly_income"] = _money(users["yearly_income"])
    df = feats.merge(users[["id", "yearly_income", "current_age", "gender"]],
                     left_on="client_id", right_on="id")
    feat_cols = [c for c in feats.columns if c != "client_id"]
    X = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = np.log1p(df["yearly_income"])
    print(f"\n모델링 테이블: N={len(df)} 사용자 × {len(feat_cols)} 행동 피처")
    print(f"소득 분포: 중앙값 ${df.yearly_income.median():,.0f} / "
          f"평균 ${df.yearly_income.mean():,.0f}")

    def hgbm(): return HistGradientBoostingRegressor(random_state=SEED, max_iter=400,
                                                     learning_rate=0.05, max_leaf_nodes=31)
    results = {"n_users": int(len(df)), "n_features": len(feat_cols)}

    # --- 베이스라인 사다리 ---
    print("\n" + "=" * 64)
    print("베이스라인 대비 성능 (5-fold CV)")
    print("=" * 64)
    ladder = {
        "mean_only": (LinearRegression, X[["n_tx"]] * 0 + 1),     # 절편만 → R²≈0
        "total_spend_only": (hgbm, X[["total_spend"]]),
        "full_behavioral": (hgbm, X),
    }
    for name, (fn, Xs) in ladder.items():
        mk = (lambda f=fn: f()) if name != "mean_only" else (lambda: LinearRegression())
        results[name] = cv_r2_ci(mk, Xs, y)
        r = results[name]
        print(f"  {name:18s} R²={r['r2']:+.3f} {r['r2_ci95']} | "
              f"중앙오차 ${r['median_abs_err_usd']:,.0f} | ±20% 적중 {r['within_20pct']*100:.1f}%")
    base = results["total_spend_only"]["r2"]
    full = results["full_behavioral"]["r2"]
    print(f"\n  → 행동 풍부함의 가치: '총지출만' R² {base:.3f} → '전체 행동' {full:.3f} "
          f"(+{(full-base):.3f}, 상대 +{(full/base-1)*100:.0f}%)")

    # --- 고소득(상위 25%) 분류 readout (의사결정 관점) ---
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score
    hi = (df["yearly_income"] >= df["yearly_income"].quantile(0.75)).astype(int)
    proba = cross_val_predict(HistGradientBoostingRegressor(random_state=SEED, max_iter=300),
                              X, hi, cv=5, method="predict")
    auc = roc_auc_score(hi, proba)
    results["high_earner_auc"] = round(float(auc), 4)
    print(f"\n고소득(상위25%) 식별 AUC: {auc:.3f}  (신용한도 배정 등 의사결정용)")

    # --- 설명가능성: permutation importance (인사이트) ---
    print("\n" + "=" * 64)
    print("어떤 행동이 소득을 추정하나 (permutation importance, 상위 12)")
    print("=" * 64)
    m = hgbm(); m.fit(X, y)
    pi = permutation_importance(m, X, y, n_repeats=10, random_state=SEED, scoring="r2")
    imp = (pd.Series(pi.importances_mean, index=feat_cols).sort_values(ascending=False))
    results["top_features"] = {k: round(float(v), 4) for k, v in imp.head(12).items()}
    for k, v in imp.head(12).items():
        print(f"  {k:20s} {v:+.4f}")

    # --- 세그먼트 공정성: 연령/성별 편향 ---
    print("\n" + "=" * 64)
    print("세그먼트 공정성 (예측 소득 편향, $ = 예측−실제 중앙값)")
    print("=" * 64)
    from sklearn.model_selection import cross_val_predict as cvp
    yhat = np.expm1(cvp(HistGradientBoostingRegressor(random_state=SEED, max_iter=400,
                        learning_rate=0.05), X, y, cv=5))
    df["_bias"] = yhat - df["yearly_income"]
    seg = {}
    df["age_band"] = pd.cut(df["current_age"], [0, 35, 50, 65, 200],
                            labels=["~35", "35-50", "50-65", "65+"])
    for col, grp in [("gender", df.groupby("gender")), ("age_band", df.groupby("age_band", observed=True))]:
        for name, sub in grp:
            seg[f"{col}:{name}"] = {"median_bias_usd": round(float(sub["_bias"].median()), 0),
                                    "n": int(len(sub))}
            print(f"  {col}:{str(name):8s} 편향 ${sub['_bias'].median():+,.0f} (n={len(sub)})")
    results["segment_bias"] = seg

    # --- thin-file 분석: 이력 길이별 정확도 ---
    if not args.quick:
        print("\n" + "=" * 64)
        print("thin-file 분석: 거래 이력이 짧으면 정확도는? (실사용 핵심)")
        print("=" * 64)
        tx_sorted = tx.sort_values(["client_id", "date"])
        seq = tx_sorted.groupby("client_id").cumcount()
        thin = {}
        for cap in [25, 50, 100, 300, 1000]:
            sub = tx_sorted[seq < cap]
            fsub = build_features(sub)
            dsub = fsub.merge(users[["id", "yearly_income"]], left_on="client_id", right_on="id")
            # 짧은 이력에선 일부 카테고리 share 컬럼이 없을 수 있어 정렬·보강
            Xs = dsub.reindex(columns=feat_cols).replace([np.inf, -np.inf], np.nan).fillna(0)
            ys = np.log1p(dsub["yearly_income"])
            r = cv_r2_ci(hgbm, Xs, ys, n_boot=100)
            thin[f"first_{cap}_tx"] = r["r2"]
            print(f"  첫 {cap:>4}건만 사용 → R²={r['r2']:+.3f} | 중앙오차 ${r['median_abs_err_usd']:,.0f}")
        results["thin_file_r2"] = thin

    REPORTS_DIR.mkdir(exist_ok=True)
    out = REPORTS_DIR / "income_estimation_results.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n결과 저장: {out}")
    print("✅ 소득 추정 파이프라인 완료")


if __name__ == "__main__":
    main()
