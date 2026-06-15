#!/usr/bin/env python3
"""
추천 시스템 정직한 재평가 (Honest Re-evaluation)
================================================

기존 `day4_evaluation.py` 의 평가에는 결과 수치를 신뢰할 수 없게 만드는
구조적 결함이 있었다. 이 모듈은 그것들을 바로잡고, 방어 가능한
신뢰할 수 있는 수치를 만든다.

기존 평가의 문제 → 이 모듈의 해결
  1. CF 평가가 정답을 후보군에서 제거 → precision/recall 이 구조적으로 항상 0
       해결: 정답(held-out)을 후보군에 포함하는 표준 leave-N-out 프로토콜
  2. CF 자리에서 실제로는 popularity 를 측정 (CF 아님)
       해결: 모델을 명시적으로 분리하고 popularity 는 '베이스라인'으로 보고
  3. seed 미지정 → 매 실행 수치가 바뀜(재현 불가)
       해결: 전 구간 고정 seed
  4. baseline 부재 → 절댓값(P@5=0.12)이 좋은지 나쁜지 판단 불가
       해결: Random / Most-Popular 베이스라인 대비 '향상(lift)' 보고
  5. 사용자 30~50명만 평가 → 통계적 노이즈, 신뢰구간 없음
       해결: 전 사용자 평가 + 부트스트랩 95% 신뢰구간
  6. `nrows=` 앞부분 샘플을 전체 데이터인 양 보고
       해결: 전체 집계 상호작용(user_merchant_interactions.csv) 사용

평가 프로토콜
  - 각 사용자의 상호작용 중 일부(기본 20%, 최소 1개)를 무작위(seed 고정)로
    test 로 가린다.
  - 남은 train 으로 모델을 학습/점수화한다.
  - train 에 있던 아이템을 제외한 전체 아이템을 순위화한다(sampled-negative 가
    아닌 all-unranked-items 프로토콜 → 더 보수적/정직).
  - HR@K, NDCG@K, Precision@K, Recall@K 를 사용자별로 계산해 평균낸다.

핵심 인사이트(보고용)
  희소 implicit 데이터에서 '인기도'는 매우 강력한 베이스라인이며, 개인화
  모델의 진짜 가치는 절댓값이 아니라 '인기도 대비 향상폭'으로 봐야 한다.

Usage:
    python src/recommendation_eval.py
    python src/recommendation_eval.py --max-users 400   # 빠른 점검
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF, TruncatedSVD

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
SEED = 42
KS = [5, 10, 20]
TEST_FRAC = 0.2


# --------------------------------------------------------------------------- #
# 데이터 → 희소 행렬
# --------------------------------------------------------------------------- #
def load_interactions(max_users: int | None = None):
    """전체 집계 상호작용을 사용자-가맹점 희소 행렬로 로드."""
    df = pd.read_csv(DATA_DIR / "user_merchant_interactions.csv")
    if max_users is not None:
        keep = df["client_id"].drop_duplicates().head(max_users)
        df = df[df["client_id"].isin(keep)]

    users = np.sort(df["client_id"].unique())
    items = np.sort(df["merchant_id"].unique())
    u_idx = {u: i for i, u in enumerate(users)}
    i_idx = {m: i for i, m in enumerate(items)}

    rows = df["client_id"].map(u_idx).to_numpy()
    cols = df["merchant_id"].map(i_idx).to_numpy()
    # implicit '신뢰도': 거래 횟수 (log 로 완화)
    vals = np.log1p(df["transaction_count"].to_numpy().astype(float))

    n_u, n_m = len(users), len(items)
    full = csr_matrix((vals, (rows, cols)), shape=(n_u, n_m))
    sparsity = 100 * (1 - full.nnz / (n_u * n_m))
    print(f"상호작용 행렬: {n_u:,} 사용자 × {n_m:,} 가맹점 | "
          f"비0 {full.nnz:,} | 희소도 {sparsity:.3f}%")
    return full, n_u, n_m


def split_train_test(full: csr_matrix, n_u: int):
    """사용자별 leave-N-out 분할 (seed 고정). test 는 가려진 정답 인덱스 집합."""
    rng = np.random.default_rng(SEED)
    train = full.tolil(copy=True)
    test_items = [set() for _ in range(n_u)]
    eligible = 0
    for u in range(n_u):
        item_idx = full[u].indices
        if len(item_idx) < 3:          # 너무 적게 거래한 사용자는 제외
            continue
        eligible += 1
        n_hold = max(1, int(len(item_idx) * TEST_FRAC))
        held = rng.choice(item_idx, size=n_hold, replace=False)
        for j in held:
            train[u, j] = 0
            test_items[u].add(int(j))
    print(f"평가 대상 사용자: {eligible:,} (상호작용 3개 이상)")
    return train.tocsr(), test_items


# --------------------------------------------------------------------------- #
# 모델: 사용자별 아이템 점수 벡터를 반환
# --------------------------------------------------------------------------- #
def score_popularity(train: csr_matrix):
    pop = np.asarray(train.sum(axis=0)).ravel()
    return lambda u: pop


def score_random(train: csr_matrix, n_m: int):
    rng = np.random.default_rng(SEED + 1)
    fixed = rng.random(n_m)        # 사용자 무관 고정 난수 (재현 가능)
    return lambda u: fixed


def score_svd(train: csr_matrix, n_components=50):
    svd = TruncatedSVD(n_components=n_components, random_state=SEED)
    uf = svd.fit_transform(train)          # (users × k)
    comp = svd.components_                  # (k × items)
    return lambda u: uf[u] @ comp


def score_nmf(train: csr_matrix, n_components=20):
    nmf = NMF(n_components=n_components, init="nndsvd", random_state=SEED, max_iter=300)
    uf = nmf.fit_transform(train)
    comp = nmf.components_
    return lambda u: uf[u] @ comp


# --------------------------------------------------------------------------- #
# 평가
# --------------------------------------------------------------------------- #
def dcg(hits: np.ndarray) -> float:
    return float(np.sum(hits / np.log2(np.arange(2, len(hits) + 2))))


def evaluate(score_fn, train: csr_matrix, test_items, n_m: int):
    """사용자별 metric 리스트를 모아 반환 (신뢰구간 계산용)."""
    per_user = {f"{m}@{k}": [] for m in ("hr", "ndcg", "precision", "recall") for k in KS}
    recommended_pool = set()
    maxk = max(KS)

    for u in range(len(test_items)):
        truth = test_items[u]
        if not truth:
            continue
        scores = np.asarray(score_fn(u), dtype=float).copy()
        scores[train[u].indices] = -np.inf        # train 아이템 제외
        # 상위 maxk 추출
        top = np.argpartition(-scores, maxk)[:maxk]
        top = top[np.argsort(-scores[top])]
        recommended_pool.update(top[:10].tolist())

        for k in KS:
            rec_k = top[:k]
            hit_mask = np.array([1.0 if i in truth else 0.0 for i in rec_k])
            n_hit = hit_mask.sum()
            ideal = dcg(np.ones(min(len(truth), k)))
            per_user[f"hr@{k}"].append(1.0 if n_hit > 0 else 0.0)
            per_user[f"ndcg@{k}"].append(dcg(hit_mask) / ideal if ideal > 0 else 0.0)
            per_user[f"precision@{k}"].append(n_hit / k)
            per_user[f"recall@{k}"].append(n_hit / len(truth))

    coverage = len(recommended_pool) / n_m
    return per_user, coverage


def summarize(per_user: dict, coverage: float):
    rng = np.random.default_rng(SEED + 2)
    out = {"coverage@10": round(coverage, 5)}
    for key, vals in per_user.items():
        arr = np.asarray(vals)
        mean = float(arr.mean()) if len(arr) else 0.0
        # 부트스트랩 95% 신뢰구간
        if len(arr) > 1:
            boot = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(500)]
            lo, hi = np.percentile(boot, [2.5, 97.5])
        else:
            lo = hi = mean
        out[key] = {"mean": round(mean, 4), "ci95": [round(float(lo), 4), round(float(hi), 4)]}
    return out


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-users", type=int, default=None,
                    help="평가 사용자 수 제한(빠른 점검용). 기본: 전체")
    args = ap.parse_args()

    print("=" * 64)
    print("추천 시스템 정직한 재평가")
    print("=" * 64)
    full, n_u, n_m = load_interactions(args.max_users)
    train, test_items = split_train_test(full, n_u)

    models = {
        "random": score_random(train, n_m),
        "popularity": score_popularity(train),
        "svd": score_svd(train),
        "nmf": score_nmf(train),
    }

    results = {}
    for name, fn in models.items():
        print(f"\n[{name}] 평가 중...")
        per_user, cov = evaluate(fn, train, test_items, n_m)
        results[name] = summarize(per_user, cov)
        print(f"  HR@10 {results[name]['hr@10']['mean']:.4f} | "
              f"NDCG@10 {results[name]['ndcg@10']['mean']:.4f} | "
              f"Coverage {cov:.4f}")

    # --- 인기도 대비 향상(lift) ---
    base = results["popularity"]["ndcg@10"]["mean"]
    print("\n" + "=" * 64)
    print("인기도(popularity) 베이스라인 대비 NDCG@10 향상(lift)")
    print("=" * 64)
    lift = {}
    for name, r in results.items():
        v = r["ndcg@10"]["mean"]
        pct = (v / base - 1) * 100 if base > 0 else 0.0
        lift[name] = round(pct, 1)
        print(f"  {name:12s} NDCG@10 {v:.4f}  ({pct:+.1f}% vs popularity)")
    results["_lift_vs_popularity_ndcg@10_pct"] = lift

    REPORTS_DIR.mkdir(exist_ok=True)
    out = REPORTS_DIR / "recommendation_eval.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n결과 저장: {out}")


if __name__ == "__main__":
    main()
