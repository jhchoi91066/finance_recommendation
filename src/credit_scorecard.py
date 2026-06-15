#!/usr/bin/env python3
"""
신용 스코어카드 (Credit Scorecard) — 은행이 실제 쓰는 점수표
============================================================

해석가능 로지스틱이 트리와 거의 동등(격차 0.01)함을 확인했으니, 이를 **은행 표준
스코어카드**로 변환한다. WOE(Weight of Evidence) 비닝 + IV(Information Value)로 피처를
고르고, 로지스틱 계수를 PDO 스케일로 *점수*로 바꾼다. 결과물은 "신청 항목별 점수표"와
"점수→연체율" 매핑 — FICO 같은 실제 여신 심사 도구의 형태다.

왜 스코어카드인가 (금융 도메인)
  - 규제(ECOA/adverse action·모델리스크 SR 11-7)는 *설명 가능한* 모델을 선호한다.
  - "왜 떨어졌나"를 항목별 점수로 고객·심사역에게 제시할 수 있다.
  - 점수 구간별 연체율이 단조롭게 정렬되면 운영 컷오프를 정하기 쉽다.

방법: WOE = ln(%정상/%연체), IV로 피처 선택 → WOE 로지스틱 → PDO 점수화
  factor = PDO/ln(2), offset = base - factor·ln(base_odds);  점수↑ = 위험↓

Usage: python src/credit_scorecard.py
"""

import json
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
REP = ROOT / "reports"
FIG = REP / "figures"
SEED = 42
NUM = ["annual_inc", "dti", "revol_util", "term_months", "emp_years",
       "open_acc", "loan_to_income", "pay_to_income"]
CAT = ["home_ownership", "purpose"]
BAD = {"Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off"}
GOOD = {"Fully Paid", "Does not meet the credit policy. Status:Fully Paid"}
# PDO 스케일
PDO, BASE, BASE_ODDS = 20, 600, 20
FACTOR = PDO / np.log(2)
OFFSET = BASE - FACTOR * np.log(BASE_ODDS)


def _pct(s):
    return pd.to_numeric(s.astype(str).str.replace("%", "", regex=False), errors="coerce")


def load():
    cols = ["loan_status", "issue_d", "loan_amnt", "annual_inc", "dti", "open_acc",
            "revol_util", "term", "emp_length", "installment", "home_ownership", "purpose"]
    df = pd.read_csv(DATA, usecols=cols, low_memory=False)
    df = df[df["loan_status"].isin(BAD | GOOD)].copy()
    df["default"] = df["loan_status"].isin(BAD).astype(int)
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df["term_months"] = _pct(df["term"].astype(str).str.replace("months", "", regex=False))
    df["revol_util"] = _pct(df["revol_util"])
    emp_map = {"< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
               "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9, "10+ years": 10}
    df["emp_years"] = df["emp_length"].map(emp_map)
    inc = df["annual_inc"].clip(lower=1)
    df["loan_to_income"] = df["loan_amnt"] / inc
    df["pay_to_income"] = (df["installment"] * 12) / inc
    return df.dropna(subset=["issue_d"]).sort_values("issue_d").reset_index(drop=True)


def num_edges(s, bins=5):
    e = np.unique(np.nanquantile(s, np.linspace(0, 1, bins + 1)))
    e[0], e[-1] = -np.inf, np.inf
    return e


def to_bin(s, feat, is_cat, edges=None, rare=None):
    if is_cat:
        g = s.astype(str).fillna("NA")
        return g.where(~g.isin(rare), "기타")
    b = pd.cut(s, bins=edges, include_lowest=True).astype(str)
    return b.where(s.notna(), "결측")


def woe_iv(binned, y):
    """bin별 WOE/IV (라플라스 평활)."""
    tg, tb = (y == 0).sum(), (y == 1).sum()
    rows = {}
    iv = 0.0
    for b, m in y.groupby(binned.values):
        g = (m == 0).sum() + 0.5
        bd = (m == 1).sum() + 0.5
        dg, db = g / tg, bd / tb
        woe = np.log(dg / db)
        rows[b] = woe
        iv += (dg - db) * woe
    return rows, float(iv)


def main():
    print("=" * 64)
    print("신용 스코어카드 (WOE/IV + PDO 점수화)")
    print("=" * 64)
    df = load()
    split = int(len(df) * 0.8)
    tr, te = df.iloc[:split], df.iloc[split:]
    print(f"학습 {len(tr):,} / 검증 {len(te):,} | 연체율 {df['default'].mean()*100:.1f}%")

    # --- 1) 비닝 + WOE/IV, 피처 선택(IV>0.02) ---
    spec = {}      # feat -> dict(is_cat, edges/rare, woe_map, iv)
    print("\n[피처 IV (예측력)]")
    for feat in NUM + CAT:
        is_cat = feat in CAT
        if is_cat:
            vc = tr[feat].astype(str).fillna("NA").value_counts()
            rare = vc[vc < 0.02 * len(tr)].index.tolist()
            edges = None
            b_tr = to_bin(tr[feat], feat, True, rare=rare)
        else:
            edges = num_edges(tr[feat])
            rare = None
            b_tr = to_bin(tr[feat], feat, False, edges=edges)
        wmap, iv = woe_iv(b_tr, tr["default"])
        spec[feat] = dict(is_cat=is_cat, edges=edges, rare=rare, woe=wmap, iv=iv)
        print(f"  {feat:16s} IV={iv:.3f}  {'✓선택' if iv>0.02 else '✗제외(약함)'}")
    used = [f for f in spec if spec[f]["iv"] > 0.02]

    # --- 2) WOE 변환 후 로지스틱(타깃=정상=1) ---
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    def woe_matrix(part):
        X = pd.DataFrame(index=part.index)
        for f in used:
            b = to_bin(part[f], f, spec[f]["is_cat"], spec[f]["edges"], spec[f]["rare"])
            med = np.median(list(spec[f]["woe"].values()))
            X[f] = b.map(spec[f]["woe"]).fillna(med).values
        return X

    Xtr, Xte = woe_matrix(tr), woe_matrix(te)
    ygood_tr = 1 - tr["default"]
    lr = LogisticRegression(max_iter=1000)
    lr.fit(Xtr, ygood_tr)
    beta = dict(zip(used, lr.coef_[0]))
    alpha = float(lr.intercept_[0])

    # --- 3) 점수화: 점수↑ = 위험↓ ---
    n = len(used)

    def score(part_woe):
        s = OFFSET + FACTOR * (alpha + sum(beta[f] * part_woe[f] for f in used))
        return s

    sc_te = score(Xte)
    auc = roc_auc_score(te["default"], -sc_te)   # 점수 낮을수록 위험
    print(f"\n스코어카드 AUC(test) = {auc:.4f}  | 점수 범위 {sc_te.min():.0f}~{sc_te.max():.0f} "
          f"(평균 {sc_te.mean():.0f})")

    # --- 4) 점수표(항목·구간별 점수) ---
    print("\n[스코어카드 — 항목별 점수표 (점수↑=위험↓)]")
    card = {}
    for f in used:
        card[f] = {}
        for b, w in sorted(spec[f]["woe"].items(), key=lambda x: x[1]):
            pts = FACTOR * beta[f] * w + (FACTOR * alpha + OFFSET) / n
            card[f][str(b)] = round(float(pts), 1)
        top = sorted(card[f].items(), key=lambda x: x[1])
        print(f"  {f} (IV {spec[f]['iv']:.2f}): "
              f"최저 {top[0][0]}={top[0][1]:.0f}점 → 최고 {top[-1][0]}={top[-1][1]:.0f}점")

    # --- 5) 점수 구간별 실제 연체율 (검증: 단조성) ---
    te2 = te.copy(); te2["score"] = sc_te.values
    te2["band"] = pd.qcut(te2["score"], 10, duplicates="drop")
    band = te2.groupby("band", observed=True)["default"].agg(["mean", "size"])
    print("\n[점수 구간별 연체율 — 단조 감소면 좋은 카드]")
    bands_out = {}
    for b, r in band.iterrows():
        bar = "█" * int(r["mean"] * 80)
        print(f"  {str(b):28s} 연체율 {r['mean']*100:5.1f}%  {bar}")
        bands_out[str(b)] = {"default_rate": round(float(r["mean"]), 4), "n": int(r["size"])}

    # --- 저장 + 차트 ---
    out = {"auc": round(float(auc), 4), "pdo": PDO, "base_score": BASE,
           "features_used": used, "iv": {f: round(spec[f]["iv"], 3) for f in spec},
           "intercept": round(alpha, 4), "scorecard_points": card,
           "score_band_default": bands_out}
    REP.mkdir(exist_ok=True)
    (REP / "credit_scorecard.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    FIG.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    # 점수 분포: 정상 vs 연체
    ax[0].hist(sc_te[te["default"].values == 0], bins=40, alpha=0.6, label="정상", color="#2e7d54", density=True)
    ax[0].hist(sc_te[te["default"].values == 1], bins=40, alpha=0.6, label="연체", color="#d64545", density=True)
    ax[0].set_title(f"신용점수 분포 (AUC {auc:.3f})", fontweight="bold")
    ax[0].set_xlabel("신용점수 (높을수록 안전)"); ax[0].legend()
    # 구간별 연체율
    centers = [iv.left + (iv.right - iv.left) / 2 for iv in band.index]
    ax[1].plot(centers, band["mean"].values * 100, "o-", color="#1a73e8", lw=2)
    ax[1].set_title("점수 구간별 실제 연체율 (단조 감소 = 좋은 카드)", fontweight="bold")
    ax[1].set_xlabel("신용점수"); ax[1].set_ylabel("연체율 (%)"); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "06_scorecard.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n저장: reports/credit_scorecard.json, reports/figures/06_scorecard.png")
    print("✅ 스코어카드 완료")


if __name__ == "__main__":
    main()
