#!/usr/bin/env python3
"""
소득 추정의 일반화·외부 타당성 검증 (Generalization / External Validity)
======================================================================

핵심 약점: 합성 데이터에서 나온 R² 0.56 이 *다른 분포*에서도 유지되는가?
완벽한 외부검증은 '거래행동+실제소득'을 모두 가진 실데이터가 공개돼 있지 않아 불가능하다.
대신 우리 데이터 안에서 **분포 이동(distribution shift)** 을 인위적으로 만들어,
"모집단이 달라지면 관계가 무너지는가"를 정직하게 측정한다.

  (1) 무작위 5-fold CV  — 같은 분포의 새 사용자 (기존 0.56 재현, 일반화 기준선)
  (2) 분포 이동: 저소비층 학습 → 고소비층 예측 (및 반대)
  (3) 분포 이동: 젊은층 학습 → 노년층 예측 (및 반대)
  (4) 분포 이동: 서부 거주 학습 → 동부 예측 (및 반대)
  (5) Ablation: 최상위 피처 3개를 제거해도 신호가 살아남는가

해석
  - 이동 테스트 R² 가 양수로 유지 → 관계가 모집단을 넘어 견고(다른 데이터로 이전 가능성 ↑)
  - 0 이나 음수로 붕괴       → 관계가 모집단 특이적(외부 일반화 약함, 정직한 한계)

Usage: python src/income_validation.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from income_estimation import _money, build_features, load_transactions

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
SEED = 42


def model():
    return HistGradientBoostingRegressor(random_state=SEED, max_iter=400,
                                         learning_rate=0.05, max_leaf_nodes=31)


def r2(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot


def cv_r2(X, y):
    kf = KFold(5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for tr, va in kf.split(X):
        m = model(); m.fit(X.iloc[tr], y.iloc[tr]); oof[va] = m.predict(X.iloc[va])
    return r2(y.to_numpy(), oof)


def shift_r2(X, y, train_mask, test_mask):
    """train 분포에서 학습 → 다른 분포(test)를 예측한 out-of-sample R²."""
    m = model()
    m.fit(X[train_mask], y[train_mask])
    pred = m.predict(X[test_mask])
    return r2(y[test_mask].to_numpy(), pred)


def main():
    print("=" * 64)
    print("소득 추정 일반화 검증")
    print("=" * 64)
    tx = load_transactions()
    feats = build_features(tx)

    users = pd.read_csv(DATA_DIR / "users_data.csv")
    users["yearly_income"] = _money(users["yearly_income"])
    df = feats.merge(users[["id", "yearly_income", "current_age", "latitude", "longitude"]],
                     left_on="client_id", right_on="id").reset_index(drop=True)
    feat_cols = [c for c in feats.columns if c != "client_id"]
    X = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = np.log1p(df["yearly_income"])

    out = {}

    # (1) 기준선: 같은 분포 새 사용자
    base = cv_r2(X, y)
    out["random_cv"] = round(float(base), 4)
    print(f"\n[기준] 무작위 5-fold CV (같은 분포 새 사용자): R²={base:+.3f}")

    print("\n[분포 이동 테스트] '학습 집단 → 다른 집단 예측'")
    shifts = {}

    # (2) 소비 수준 이동
    med_spend = df["total_spend"].median()
    low, high = df["total_spend"] < med_spend, df["total_spend"] >= med_spend
    shifts["low_spend→high_spend"] = shift_r2(X, y, low, high)
    shifts["high_spend→low_spend"] = shift_r2(X, y, high, low)

    # (3) 연령 이동
    med_age = df["current_age"].median()
    young, old = df["current_age"] < med_age, df["current_age"] >= med_age
    shifts["young→old"] = shift_r2(X, y, young, old)
    shifts["old→young"] = shift_r2(X, y, old, young)

    # (4) 지역 이동 (거주 경도 기준 서/동)
    med_lon = df["longitude"].median()
    west, east = df["longitude"] < med_lon, df["longitude"] >= med_lon
    shifts["west→east"] = shift_r2(X, y, west, east)
    shifts["east→west"] = shift_r2(X, y, east, west)

    for k, v in shifts.items():
        drop = (v / base - 1) * 100 if base else 0
        flag = "✅견고" if v > 0.30 else ("⚠️약화" if v > 0.10 else "❌붕괴")
        print(f"  {k:22s} R²={v:+.3f}  (기준 대비 {drop:+.0f}%)  {flag}")
    out["distribution_shift"] = {k: round(float(v), 4) for k, v in shifts.items()}

    # (5) Ablation: 최상위 피처 제거
    print("\n[Ablation] 최상위 피처를 빼도 신호가 살아남는가")
    top3 = ["std_amt", "n_states", "merchants_per_year"]
    Xabl = X.drop(columns=[c for c in top3 if c in X.columns])
    abl = cv_r2(Xabl, y)
    out["ablation_drop_top3"] = round(float(abl), 4)
    print(f"  상위3({', '.join(top3)}) 제거 → R²={abl:+.3f} "
          f"(전체 {base:+.3f} 대비 {(abl/base-1)*100:+.0f}%)")

    # 요약 판정
    shift_vals = list(shifts.values())
    worst = min(shift_vals)
    verdict = ("분포가 달라져도 관계가 견고하게 유지됨 → 외부 일반화 가능성 높음"
               if worst > 0.30 else
               "일부 분포 이동에서 성능이 약화/붕괴 → 모집단 특이적 신호 존재(정직한 한계)")
    out["verdict"] = verdict
    print(f"\n판정: 최악 이동 R²={worst:+.3f} → {verdict}")

    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "income_validation_results.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    print("\n결과 저장: reports/income_validation_results.json")


if __name__ == "__main__":
    main()
