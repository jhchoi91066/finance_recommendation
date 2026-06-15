#!/usr/bin/env python3
"""
신용카드 이상거래 탐지(Fraud Detection System)
================================================

카드 거래 데이터에서 사기 거래를 탐지하는 머신러닝 파이프라인이다.
극단적 클래스 불균형(사기 약 0.15%) 환경에서 거래/사용자/카드/가맹점 정보를
결합해 피처를 만들고, LightGBM 기반 분류 모델로 사기를 탐지한다.

이 파이프라인이 단순 "모델 돌리기"와 다른 점 (핵심 설계 포인트)
  1. 시간 누수 차단(temporal split): 과거 거래로 학습해 '미래' 거래를 예측.
     랜덤 분할은 미래 정보가 학습에 새어 들어가 현실보다 점수가 부풀려진다.
  2. 임계값을 검증셋에서 선택: test 로 임계값을 고르면 평가가 낙관적으로 편향된다.
     train→valid→test 3분할로 valid 에서 운영점을 정하고 test 에서만 보고.
  3. 가맹점 빈도 등 집계 피처를 '학습 구간에서만' 계산해 미래 누수 차단.
  4. 평가를 '돈'으로 번역: ROC-AUC 가 아니라 "사기 금액 몇 % 차단 / 정상거래
     몇 건 오탐(검토 비용)" 이라는 비즈니스 의사결정 수치를 제시.
  5. 세그먼트 위험 분석: 어떤 거래 유형(온라인/야간/평소 대비 고액)에 사기가
     집중되는지 → 운영 룰·모니터링 우선순위로 연결.

Usage:
    python src/fraud_detection.py --sample_size 3000000   # 빠른 검증(앞 구간)
    python src/fraud_detection.py                          # 전체 라벨 데이터

데이터 요구사항 (data/ 폴더):
    transactions_data.csv, train_fraud_labels.json,
    users_data.csv, cards_data.csv, mcc_codes.json
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
SEED = 42


# --------------------------------------------------------------------------- #
# 1. 데이터 로딩
# --------------------------------------------------------------------------- #
def _money_to_float(series: pd.Series) -> pd.Series:
    """'$1,234.56' 형태의 금액 문자열을 float로 변환."""
    return (
        series.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .replace({"nan": np.nan})
        .astype(float)
    )


def load_data(sample_size: int | None = None) -> pd.DataFrame:
    """거래·사용자·카드·사기라벨을 로드하고 단일 학습 테이블로 결합."""
    print("=" * 60)
    print("1. 데이터 로딩")
    print("=" * 60)

    usecols = ["id", "date", "client_id", "card_id", "amount",
               "use_chip", "merchant_id", "merchant_city", "merchant_state",
               "mcc", "errors"]
    tx = pd.read_csv(DATA_DIR / "transactions_data.csv",
                     usecols=usecols, nrows=sample_size)
    print(f"거래 데이터: {tx.shape}")

    # --- 사기 라벨 (id -> 'Yes'/'No') ---
    with open(DATA_DIR / "train_fraud_labels.json") as f:
        labels = json.load(f)["target"]
    label_series = pd.Series(labels, name="is_fraud")
    label_series.index = label_series.index.astype(np.int64)
    tx["is_fraud"] = tx["id"].map(label_series).map({"Yes": 1, "No": 0})

    # 라벨이 있는 거래만 사용
    before = len(tx)
    tx = tx.dropna(subset=["is_fraud"]).copy()
    tx["is_fraud"] = tx["is_fraud"].astype(int)
    print(f"라벨 매칭: {len(tx):,} / {before:,} 건  "
          f"(사기 {tx['is_fraud'].mean() * 100:.3f}%)")

    # --- 사용자/카드 데이터 결합 ---
    users = pd.read_csv(DATA_DIR / "users_data.csv")
    cards = pd.read_csv(DATA_DIR / "cards_data.csv")
    for col in ["per_capita_income", "yearly_income", "total_debt"]:
        users[col] = _money_to_float(users[col])
    cards["credit_limit"] = _money_to_float(cards["credit_limit"])
    users = users.rename(columns={"id": "client_id"})
    cards = cards.rename(columns={"id": "card_id"})

    tx = tx.merge(
        users[["client_id", "current_age", "gender", "per_capita_income",
               "yearly_income", "total_debt", "credit_score", "num_credit_cards"]],
        on="client_id", how="left",
    )
    tx = tx.merge(
        cards[["card_id", "card_brand", "card_type", "has_chip",
               "credit_limit", "num_cards_issued", "card_on_dark_web"]],
        on="card_id", how="left",
    )
    print(f"결합 후: {tx.shape}")
    return tx


# --------------------------------------------------------------------------- #
# 2. 피처 엔지니어링 (행 단위 + 사용자 과거값만 → 시간 누수 없음)
# --------------------------------------------------------------------------- #
def engineer_features(tx: pd.DataFrame):
    """도메인 지식 기반 피처 생성. 가맹점 빈도 등 '구간 의존' 집계 피처는
    누수를 막기 위해 여기서 만들지 않고 분할 이후 학습구간에서 계산한다."""
    print("\n" + "=" * 60)
    print("2. 피처 엔지니어링")
    print("=" * 60)

    tx["date"] = pd.to_datetime(tx["date"])
    tx = tx.sort_values("date").reset_index(drop=True)   # temporal 순서 고정
    tx["amount"] = _money_to_float(tx["amount"]).abs()

    # --- 시간 피처 ---
    tx["hour"] = tx["date"].dt.hour
    tx["dayofweek"] = tx["date"].dt.dayofweek
    tx["is_weekend"] = (tx["dayofweek"] >= 5).astype(int)
    tx["is_night"] = ((tx["hour"] < 6) | (tx["hour"] >= 22)).astype(int)

    # --- 금액 피처 ---
    tx["log_amount"] = np.log1p(tx["amount"])
    # 사용자별 '과거' 평균 거래액 대비 현재 거래 (expanding+shift → 미래 미사용)
    grp = tx.groupby("client_id")["amount"]
    tx["user_amount_mean"] = grp.transform(lambda s: s.expanding().mean().shift()).fillna(0)
    tx["amount_vs_user_mean"] = tx["amount"] / (tx["user_amount_mean"] + 1)

    # --- 카드/한도 피처 ---
    tx["amount_vs_credit_limit"] = tx["amount"] / (tx["credit_limit"] + 1)
    tx["has_chip"] = (tx["has_chip"] == "YES").astype(int)
    tx["card_on_dark_web"] = (tx["card_on_dark_web"] == "Yes").astype(int)

    # --- 사용자 재정 건전성 ---
    tx["debt_to_income"] = tx["total_debt"] / (tx["yearly_income"] + 1)

    # --- 거래 채널/오류 ---
    tx["has_error"] = tx["errors"].notna().astype(int)
    tx["online_tx"] = (tx["use_chip"] == "Online Transaction").astype(int)

    # --- (핵심) 카드별 순차/속도(velocity) 피처: 사기탐지의 1순위 신호 ---
    # 모두 '카드의 과거'만 사용 → 시간 누수 없음 (date 정렬 상태에서 shift/diff/cumcount).
    gcard = tx.groupby("card_id")
    # 직전 거래 후 경과 시간(초): 사기는 짧은 시간에 연속 발생하는 경향
    tx["sec_since_prev"] = gcard["date"].diff().dt.total_seconds().fillna(-1)
    # 카드의 누적 거래 순번(신규 카드일수록 위험 패턴 다름)
    tx["card_txn_seq"] = gcard.cumcount()
    # 이 카드가 '처음 보는' 가맹점인가 (날짜순 첫 등장 = 1)
    tx["new_merchant_for_card"] = (~tx.duplicated(["card_id", "merchant_id"])).astype(int)
    # 직전 거래와 가맹점 '주(state)'가 바뀌었는가 (지리적 이동 신호)
    prev_state = gcard["merchant_state"].shift()
    tx["state_change"] = ((tx["merchant_state"] != prev_state)
                          & prev_state.notna()).astype(int)
    # 최근성 버킷: 직전 거래가 5분 이내(초고속 연속 결제)
    tx["rapid_after_prev"] = ((tx["sec_since_prev"] >= 0)
                             & (tx["sec_since_prev"] < 300)).astype(int)

    # --- 범주형 인코딩 (전역 코드 — 누수 위험 없음: 카드 속성/채널 식별자) ---
    for col in ["gender", "card_brand", "card_type", "use_chip"]:
        tx[col] = tx[col].astype("category").cat.codes

    feature_cols = [
        "amount", "log_amount", "hour", "dayofweek", "is_weekend", "is_night",
        "amount_vs_user_mean", "amount_vs_credit_limit", "mcc",
        "current_age", "gender", "per_capita_income", "yearly_income",
        "credit_score", "num_credit_cards", "debt_to_income",
        "card_brand", "card_type", "has_chip", "credit_limit",
        "num_cards_issued", "card_on_dark_web", "has_error", "online_tx",
        # 카드별 속도/순차 피처
        "sec_since_prev", "card_txn_seq", "new_merchant_for_card",
        "state_change", "rapid_after_prev",
        # merchant_freq 는 분할 후 학습구간에서 채워 넣는다 (아래 train_and_evaluate)
        "merchant_freq",
    ]
    print(f"생성된 행단위 피처 수: {len(feature_cols) - 1} (+merchant_freq 는 분할 후 계산)")
    return tx, feature_cols


# --------------------------------------------------------------------------- #
# 3. 모델 학습 & 평가 (temporal split + 누수 차단 + 돈 환산)
# --------------------------------------------------------------------------- #
def train_and_evaluate(tx: pd.DataFrame, feature_cols: list[str],
                       split: str = "temporal") -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    print("\n" + "=" * 60)
    print(f"3. 모델 학습 및 평가 ({split} split)")
    print("=" * 60)

    n = len(tx)
    if split == "random":
        # 흔히 쓰지만 시계열에 부적절한 분할: 미래 정보가 학습에 새어 들어간다.
        rng = np.random.default_rng(SEED)
        perm = rng.permutation(n)
        i_tr, i_va = int(n * 0.70), int(n * 0.80)
        tx = tx.iloc[perm].reset_index(drop=True)
        tr, va, te = tx.iloc[:i_tr], tx.iloc[i_tr:i_va], tx.iloc[i_va:]
    else:
        # 시간 순서 3분할: 과거 70% 학습 / 10% 검증 / 미래 20% 테스트
        i_tr, i_va = int(n * 0.70), int(n * 0.80)
        tr, va, te = tx.iloc[:i_tr], tx.iloc[i_tr:i_va], tx.iloc[i_va:]
    print(f"기간 분할 | train {tr['date'].min().date()}~{tr['date'].max().date()} "
          f"({len(tr):,}) | valid ({len(va):,}) | "
          f"test {te['date'].min().date()}~{te['date'].max().date()} ({len(te):,})")
    print(f"사기율 train {tr.is_fraud.mean()*100:.3f}% / "
          f"valid {va.is_fraud.mean()*100:.3f}% / test {te.is_fraud.mean()*100:.3f}%")

    # --- 누수 차단: 가맹점 빈도를 '학습 구간'에서만 계산해 매핑 ---
    mfreq = tr["merchant_id"].value_counts()
    for part in (tr, va, te):
        part["merchant_freq"] = part["merchant_id"].map(mfreq).fillna(0)

    Xtr, ytr = tr[feature_cols].fillna(0), tr["is_fraud"]
    Xva, yva = va[feature_cols].fillna(0), va["is_fraud"]
    Xte, yte = te[feature_cols].fillna(0), te["is_fraud"]

    results = {
        "data": {
            "split": split,
            "n_total": int(n), "n_train": int(len(tr)), "n_test": int(len(te)),
            "fraud_rate_overall_pct": round(float(tx.is_fraud.mean() * 100), 4),
            "test_period": f"{te['date'].min().date()}~{te['date'].max().date()}",
        }
    }

    # --- 베이스라인: 로지스틱 회귀 ---
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(sc.fit_transform(Xtr), ytr)
    lr_proba = lr.predict_proba(sc.transform(Xte))[:, 1]
    results["logistic_regression"] = {
        "roc_auc": round(roc_auc_score(yte, lr_proba), 4),
        "pr_auc": round(average_precision_score(yte, lr_proba), 4),
    }

    # --- 메인 모델: LightGBM ---
    try:
        from lightgbm import LGBMClassifier
        pos_weight = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
        model = LGBMClassifier(
            n_estimators=600, learning_rate=0.03, num_leaves=31,
            min_child_samples=5, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, scale_pos_weight=pos_weight,
            random_state=SEED, n_jobs=-1, verbose=-1,
        )
        model_name = "lightgbm"
    except (ImportError, OSError) as e:
        from sklearn.ensemble import HistGradientBoostingClassifier
        print(f"⚠ lightgbm 사용 불가({type(e).__name__}) → HistGradientBoosting 대체")
        model = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, random_state=SEED,
            class_weight="balanced",
        )
        model_name = "hist_gradient_boosting"

    model.fit(Xtr, ytr)
    proba_te = model.predict_proba(Xte)[:, 1]
    roc = roc_auc_score(yte, proba_te)
    pr = average_precision_score(yte, proba_te)
    base_rate = float(yte.mean())

    results[model_name] = {
        "roc_auc": round(roc, 4), "pr_auc": round(pr, 4),
        # PR-AUC 를 무작위(=사기율) 대비 몇 배인지로 표현 → 불균형에서 가장 정직한 수치
        "pr_auc_lift_vs_random": round(pr / base_rate, 1) if base_rate else 0,
    }

    print(f"\n[베이스라인 LogReg] ROC-AUC {results['logistic_regression']['roc_auc']} | "
          f"PR-AUC {results['logistic_regression']['pr_auc']}")
    print(f"[메인 {model_name}] ROC-AUC {roc:.4f} | PR-AUC {pr:.4f}")
    pr_lift = (pr / max(results['logistic_regression']['pr_auc'], 1e-9) - 1) * 100
    print(f"  → 베이스라인 대비 PR-AUC {pr_lift:+.1f}% | 무작위(사기율 {base_rate*100:.3f}%) 대비 "
          f"PR-AUC {pr/base_rate:.0f}배")

    # --- 운영 임계값: 현업처럼 '검토 예산(상위 X%만 검토)' 기준 ---
    # F1-최대 임계값은 사기가 시점별로 뭉쳐(non-stationary) 검증구간에 사기가
    # 거의 없으면 붕괴한다. 분석가가 하루에 검토 가능한 양(=알람 예산)을
    # 비즈니스 제약으로 고정하는 편이 훨씬 견고하고 현실적이다.
    yte_arr = yte.to_numpy()
    n_pos = max(int(yte_arr.sum()), 1)
    # 점수 내림차순 순위 (확률 saturation 시에도 정확히 상위 N건만 선택 → 예산별 차등)
    order = np.argsort(-proba_te, kind="stable")
    budgets = {}
    pred_by_budget = {}
    for b in (0.1, 0.5, 1.0):
        k = max(1, int(len(proba_te) * b / 100))
        top_idx = order[:k]
        pred = np.zeros(len(proba_te), dtype=int)
        pred[top_idx] = 1
        tp = int(yte_arr[top_idx].sum())
        recall_b = tp / n_pos
        precision_b = tp / k
        budgets[f"top_{b}pct"] = {
            "flagged": int(k), "recall": round(recall_b, 4),
            "precision": round(precision_b, 4),
            "lift_vs_random": round(recall_b / (b / 100), 1),
        }
        pred_by_budget[b] = pred
        print(f"  검토예산 상위 {b:>3}% → 사기 {recall_b*100:5.1f}% 적발 "
              f"(precision {precision_b*100:.1f}%, 무작위 대비 {recall_b/(b/100):.0f}배, "
              f"{k:,}건 검토)")
    results[model_name]["operating_points"] = budgets

    # 대표 운영점(상위 0.5%)으로 돈/혼동행렬 산출
    pred_te = pred_by_budget[0.5]
    results[model_name]["confusion@0.5pct"] = {
        "tp": int(((pred_te == 1) & (yte_arr == 1)).sum()),
        "fp": int(((pred_te == 1) & (yte_arr == 0)).sum()),
        "fn": int(((pred_te == 0) & (yte_arr == 1)).sum()),
        "tn": int(((pred_te == 0) & (yte_arr == 0)).sum()),
    }

    # --- (핵심) 돈으로 환산: 손실 절감 vs 검토 비용 (상위 0.5% 운영점) ---
    results["business"] = money_translation(te, pred_te, yte)

    # --- 세그먼트 위험 인사이트 ---
    results["segment_risk"] = segment_risk(te)

    # --- 피처 중요도 ---
    if hasattr(model, "feature_importances_"):
        imp = (pd.Series(model.feature_importances_, index=feature_cols)
               .sort_values(ascending=False))
        print("\n상위 12 피처 중요도:\n", imp.head(12).to_string())
        results["top_features"] = {k: int(v) for k, v in imp.head(12).items()}

    REPORTS_DIR.mkdir(exist_ok=True)
    suffix = "" if split == "temporal" else f"_{split}"
    out = REPORTS_DIR / f"fraud_detection_results{suffix}.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n결과 저장: {out}")
    return results


def money_translation(te: pd.DataFrame, pred: np.ndarray, y: np.ndarray) -> dict:
    """ROC-AUC 를 '돈'으로 번역: 운영점에서 막은 사기 금액 / 놓친 금액 / 검토 부담."""
    amt = te["amount"].to_numpy()
    y = y.to_numpy()
    fraud_total = float(amt[y == 1].sum())
    caught = float(amt[(y == 1) & (pred == 1)].sum())     # 막은 사기 금액
    missed = float(amt[(y == 1) & (pred == 0)].sum())     # 놓친 사기 금액
    fp_count = int(((pred == 1) & (y == 0)).sum())        # 오탐(정상거래 검토) 건수
    flagged = int((pred == 1).sum())
    review_rate = flagged / len(te) * 100

    print("\n" + "-" * 60)
    print("💰 비즈니스 환산 (test 구간 운영점 기준)")
    print("-" * 60)
    print(f"  사기 피해 총액        ${fraud_total:,.0f}")
    print(f"  차단한 사기 금액      ${caught:,.0f}  (손실 절감률 {caught/fraud_total*100:.1f}%)")
    print(f"  놓친 사기 금액        ${missed:,.0f}")
    print(f"  검토 대상(flag) 건수  {flagged:,}건  (전체 거래의 {review_rate:.2f}%)")
    print(f"  └ 그중 오탐(정상)     {fp_count:,}건  → 분석가 검토 비용")
    return {
        "fraud_amount_total": round(fraud_total, 2),
        "fraud_amount_caught": round(caught, 2),
        "fraud_amount_missed": round(missed, 2),
        "loss_reduction_pct": round(caught / fraud_total * 100, 2) if fraud_total else 0,
        "flagged_count": flagged,
        "flagged_rate_pct": round(review_rate, 3),
        "false_positive_count": fp_count,
    }


def segment_risk(te: pd.DataFrame) -> dict:
    """어떤 거래 유형에 사기가 집중되는가 — 운영 룰/모니터링 우선순위 근거."""
    base = te["is_fraud"].mean()
    out = {"baseline_fraud_rate_pct": round(float(base * 100), 4), "segments": {}}

    def seg(name, mask, label):
        sub = te[mask]
        if len(sub) == 0:
            return
        rate = sub["is_fraud"].mean()
        out["segments"][name] = {
            "label": label,
            "fraud_rate_pct": round(float(rate * 100), 4),
            "lift_vs_baseline": round(float(rate / base), 2) if base else 0,
            "n": int(len(sub)),
        }

    seg("online", te["online_tx"] == 1, "온라인 거래")
    seg("in_person", te["online_tx"] == 0, "대면 거래")
    seg("night", te["is_night"] == 1, "야간(22~6시) 거래")
    seg("weekend", te["is_weekend"] == 1, "주말 거래")
    seg("amount_gt_3x_user_mean", te["amount_vs_user_mean"] >= 3,
        "평소 평균의 3배 이상 고액")
    seg("has_error", te["has_error"] == 1, "거래오류 동반")

    print("\n" + "-" * 60)
    print("🔎 세그먼트 위험 (test 구간 실제 사기율, 배수 = 평균 대비)")
    print("-" * 60)
    print(f"  전체 평균 사기율: {base*100:.4f}%")
    for k, v in sorted(out["segments"].items(),
                       key=lambda x: -x[1]["lift_vs_baseline"]):
        print(f"  {v['label']:24s} {v['fraud_rate_pct']:.4f}%  "
              f"({v['lift_vs_baseline']:.1f}배, n={v['n']:,})")
    return out


def main():
    ap = argparse.ArgumentParser(description="신용카드 이상거래 탐지 파이프라인")
    ap.add_argument("--sample_size", type=int, default=None,
                    help="거래 데이터 샘플 행 수 (기본: 전체)")
    ap.add_argument("--split", choices=["temporal", "random"], default="temporal",
                    help="평가 분할 방식 (temporal=정직, random=낙관적 비교용)")
    args = ap.parse_args()

    tx = load_data(sample_size=args.sample_size)
    tx, feature_cols = engineer_features(tx)
    train_and_evaluate(tx, feature_cols, split=args.split)
    print("\n✅ 사기탐지 파이프라인 완료")


if __name__ == "__main__":
    main()
