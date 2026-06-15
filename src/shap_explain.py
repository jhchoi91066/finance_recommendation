#!/usr/bin/env python3
"""
SHAP 설명가능성 — 신용위험 모델 (LightGBM 내장 정확 SHAP)
========================================================

"왜 이 대출이 위험한가"를 SHAP 값으로 설명한다. 금융은 모델리스크 관리(SR 11-7)와
adverse-action 고지 때문에 *개별 의사결정의 근거*를 요구한다 — SHAP은 그 표준 도구다.

구현 메모: `shap` 패키지 대신 **LightGBM 내장 `pred_contrib=True`** 를 사용한다.
트리모델의 정확한(exact) SHAP 값을 의존성 추가 없이 계산하며, Python 3.14에서도 안정적이다.

산출
  07_shap_summary : (좌) 전역 중요도 = 평균 |SHAP|, (우) beeswarm = 피처값↔영향 분포
  08_shap_reasons : 개별 대출 2건의 사유코드(위험을 올린/내린 요인) waterfall

Usage: python src/shap_explain.py
"""

import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
for f in ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic"]:
    try:
        matplotlib.font_manager.findfont(f, fallback_to_default=False)
        plt.rcParams["font.family"] = f; break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data_lending" / "loan.csv"
FIG = ROOT / "reports" / "figures"
SEED = 42
BAD = {"Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off"}
GOOD = {"Fully Paid", "Does not meet the credit policy. Status:Fully Paid"}
FEATS = ["loan_amnt", "annual_inc", "dti", "open_acc", "revol_util", "total_acc",
         "installment", "term_months", "emp_years", "loan_to_income", "pay_to_income",
         "inst_to_loan", "home_ownership_code", "purpose_code"]


def _pct(s):
    return pd.to_numeric(s.astype(str).str.replace("%", "", regex=False), errors="coerce")


def load():
    cols = ["loan_status", "issue_d", "loan_amnt", "annual_inc", "dti", "open_acc",
            "revol_util", "total_acc", "term", "emp_length", "installment",
            "home_ownership", "purpose"]
    df = pd.read_csv(DATA, usecols=cols, low_memory=False)
    df = df[df["loan_status"].isin(BAD | GOOD)].copy()
    df["default"] = df["loan_status"].isin(BAD).astype(int)
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df["term_months"] = _pct(df["term"].astype(str).str.replace("months", "", regex=False))
    df["revol_util"] = _pct(df["revol_util"])
    emp = {"< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4, "5 years": 5,
           "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9, "10+ years": 10}
    df["emp_years"] = df["emp_length"].map(emp)
    inc = df["annual_inc"].clip(lower=1)
    df["loan_to_income"] = df["loan_amnt"] / inc
    df["pay_to_income"] = (df["installment"] * 12) / inc
    df["inst_to_loan"] = df["installment"] / (df["loan_amnt"] + 1)
    df["home_ownership_code"] = df["home_ownership"].astype("category").cat.codes
    df["purpose_code"] = df["purpose"].astype("category").cat.codes
    return df.dropna(subset=["issue_d"]).sort_values("issue_d").reset_index(drop=True)


def main():
    from lightgbm import LGBMClassifier
    print("로딩·학습...")
    df = load()
    split = int(len(df) * 0.8)
    X = df[FEATS].apply(pd.to_numeric, errors="coerce").fillna(0)
    y = df["default"]
    m = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                       min_child_samples=50, subsample=0.9, colsample_bytree=0.7,
                       reg_lambda=5.0, random_state=SEED, n_jobs=-1, verbose=-1)
    m.fit(X.iloc[:split], y.iloc[:split])

    # 테스트 샘플의 정확 SHAP (마지막 열 = base value)
    rng = np.random.default_rng(SEED)
    te_idx = np.arange(split, len(df))
    samp = rng.choice(te_idx, size=min(30000, len(te_idx)), replace=False)
    Xs = X.iloc[samp]
    contrib = m.booster_.predict(Xs, pred_contrib=True)
    shap = contrib[:, :-1]
    base = contrib[0, -1]
    mean_abs = np.abs(shap).mean(0)
    order = np.argsort(mean_abs)[::-1]
    print("SHAP 전역 중요도 상위:")
    for i in order[:8]:
        print(f"  {FEATS[i]:18s} {mean_abs[i]:.4f}")

    # ---- 그림1: 전역 중요도 + beeswarm ----
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    top = order[:12][::-1]
    ax[0].barh([FEATS[i] for i in top], mean_abs[top], color="#1a73e8")
    ax[0].set_title("SHAP 전역 중요도 (평균 |영향|)", fontweight="bold")
    ax[0].set_xlabel("연체 로그오즈에 대한 평균 절대 기여")

    # beeswarm: 상위 10 피처, 색=피처값(백분위)
    topb = order[:10][::-1]
    for row, fi in enumerate(topb):
        sv = shap[:, fi]
        fv = Xs.iloc[:, fi].to_numpy().astype(float)
        pr = pd.Series(fv).rank(pct=True).to_numpy()
        jit = (rng.random(len(sv)) - 0.5) * 0.6
        sl = rng.choice(len(sv), size=min(2500, len(sv)), replace=False)
        sc = ax[1].scatter(sv[sl], np.full(len(sl), row) + jit[sl], c=pr[sl],
                           cmap="coolwarm", s=6, alpha=0.5)
    ax[1].set_yticks(range(len(topb))); ax[1].set_yticklabels([FEATS[i] for i in topb])
    ax[1].axvline(0, color="gray", lw=0.8)
    ax[1].set_title("SHAP beeswarm (오른쪽=위험↑) · 색: 빨강=피처값 큼", fontweight="bold")
    ax[1].set_xlabel("SHAP 값 (연체 위험 기여)")
    fig.colorbar(sc, ax=ax[1], label="피처값 백분위")
    fig.tight_layout(); fig.savefig(FIG / "07_shap_summary.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ---- 그림2: 개별 대출 사유코드 (가장 위험/안전 예측 1건씩) ----
    proba = m.predict_proba(Xs)[:, 1]
    pick = {"위험 예측 대출": int(np.argmax(proba)), "안전 예측 대출": int(np.argmin(proba))}
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax2, (title, i) in zip(axes, pick.items()):
        sv = shap[i]
        o = np.argsort(np.abs(sv))[::-1][:8][::-1]
        colors = ["#d64545" if sv[j] > 0 else "#2e7d54" for j in o]
        ax2.barh([f"{FEATS[j]}={Xs.iloc[i, j]:.2f}" for j in o], sv[o], color=colors)
        ax2.axvline(0, color="gray", lw=0.8)
        ax2.set_title(f"{title} (연체확률 {proba[i]*100:.0f}%)\n빨강=위험↑ 초록=위험↓",
                      fontweight="bold")
        ax2.set_xlabel("SHAP 기여")
    fig.tight_layout(); fig.savefig(FIG / "08_shap_reasons.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n저장: reports/figures/07_shap_summary.png, 08_shap_reasons.png")
    print("✅ SHAP 설명가능성 완료")


if __name__ == "__main__":
    main()
