#!/usr/bin/env python3
"""
프로젝트 핵심 발견 시각화 (Story-driven Visualization)
=====================================================

모델을 재실행하지 않고 `reports/*.json` 결과만 읽어, '정직한 평가'라는 서사를 담은
핵심 차트를 생성한다. 산출물: reports/figures/*.png

생성 차트
  00_dashboard          : 6개 핵심 발견 한 장 요약(히어로)
  01_credit_leakage     : 데이터 누수의 대가 (실데이터)
  02_model_development  : 고도화 ablation + 해석모델 비교 + 롤링검증
  03_income_estimation  : 소득 베이스라인 사다리 + thin-file 곡선

Usage: python src/visualize.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 한글 폰트 (macOS)
for f in ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic"]:
    try:
        matplotlib.font_manager.findfont(f, fallback_to_default=False)
        plt.rcParams["font.family"] = f
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / "reports"
FIG = REP / "figures"
FIG.mkdir(exist_ok=True)

C_BAD, C_GOOD, C_BASE, C_HI = "#d64545", "#2e7d54", "#9aa0a6", "#1a73e8"


def load(name):
    return json.loads((REP / name).read_text())


cr = load("credit_risk_results.json")
adv = load("credit_risk_advanced_results.json")
inc = load("income_estimation_results.json")
val = load("income_validation_results.json")
rec = load("recommendation_eval.json")
frd = load("fraud_detection_results.json")
frr = load("fraud_detection_results_random.json")


def annotate(ax, bars, fmt="{:.3f}", dy=0.005):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy,
                fmt.format(b.get_height()), ha="center", va="bottom", fontsize=9, fontweight="bold")


# --------- 개별 패널 그리기 함수들 (재사용) ---------
def panel_leakage(ax):
    labels = ["누수 포함\n(가짜)", "신청정보만\n(정직)", "+ LC등급"]
    vals = [cr["model_leaky"]["roc_auc"], cr["model_app_only"]["roc_auc"],
            cr["model_app_plus_lc"]["roc_auc"]]
    bars = ax.bar(labels, vals, color=[C_BAD, C_GOOD, C_HI])
    annotate(ax, bars)
    ax.axhline(0.5, ls="--", c="gray", lw=0.8); ax.set_ylim(0.4, 1.05)
    ax.set_title("① 데이터 누수의 대가 (실데이터 신용위험)", fontweight="bold")
    ax.set_ylabel("ROC-AUC")
    ax.text(0.5, 0.78, "누수의 대가\n= 0.30", ha="center", color=C_BAD, fontsize=9, fontweight="bold")


def panel_ablation(ax):
    ab = adv["ablation"]
    labels = ["M0\nv1", "M1\n+피처", "M2\n+범주형", "M3\n+튜닝"]
    vals = [ab["M0_v1재현"]["roc_auc"], ab["M1_+피처엔지니어링"]["roc_auc"],
            ab["M2_+네이티브범주형"]["roc_auc"], ab["M3_+튜닝(최종)"]["roc_auc"]]
    bars = ax.bar(labels, vals, color=C_HI)
    annotate(ax, bars, dy=0.0005)
    ax.set_ylim(0.69, 0.705)
    ax.set_title("② 모델 고도화 ablation: 이득 +0.004뿐\n→ 천장은 알고리즘 아닌 데이터", fontweight="bold")
    ax.set_ylabel("ROC-AUC")


def panel_interp(ax):
    lv = adv["logistic_vs_lgbm"]
    roll = adv["rolling_validation"]["per_year"]
    labels = ["로지스틱\n(설명가능)", "LightGBM\n(트리)"]
    bars = ax.bar(labels, [lv["logistic_auc"], lv["lgbm_auc"]], color=[C_GOOD, C_HI])
    annotate(ax, bars, dy=0.001)
    ax.set_ylim(0.6, 0.75)
    ax.set_title(f"③ 해석가능 모델이 거의 동등 (격차 {lv['gap']:.2f})\n→ 규제용 설명모델로 충분", fontweight="bold")
    ax.set_ylabel("ROC-AUC")
    # 롤링검증을 텍스트로
    rtxt = " / ".join(f"{y}:{v:.3f}" for y, v in roll.items())
    ax.text(0.5, -0.28, f"롤링검증 {rtxt}  (평균 {adv['rolling_validation']['mean']:.3f}±{adv['rolling_validation']['std']:.3f})",
            transform=ax.transAxes, ha="center", fontsize=8, color="gray")


def panel_income_ladder(ax):
    labels = ["평균\n(naive)", "총지출만", "전체 행동\n(26피처)"]
    vals = [max(inc["mean_only"]["r2"], 0), inc["total_spend_only"]["r2"], inc["full_behavioral"]["r2"]]
    bars = ax.bar(labels, vals, color=[C_BASE, C_BASE, C_GOOD])
    annotate(ax, bars, fmt="{:.2f}", dy=0.01)
    ax.set_ylim(0, 0.65)
    ax.set_title("④ 소득 추정: '얼마'보다 '어떻게'\n총지출만 0.17 → 전체행동 0.56 (+230%)", fontweight="bold")
    ax.set_ylabel("CV R²")


def panel_thinfile(ax):
    tf = inc["thin_file_r2"]
    xs = [25, 50, 100, 300, 1000, inc["n_users"] and 11000]
    ys = [tf["first_25_tx"], tf["first_50_tx"], tf["first_100_tx"],
          tf["first_300_tx"], tf["first_1000_tx"], inc["full_behavioral"]["r2"]]
    ax.plot(xs, ys, "o-", color=C_HI, lw=2)
    ax.set_xscale("log")
    ax.set_title("⑤ thin-file 역설: 신규 고객일수록 안 됨\n(첫 100건 R²0.08 → 전체 0.56)", fontweight="bold")
    ax.set_xlabel("사용한 거래 이력 수(log)"); ax.set_ylabel("CV R²")
    ax.grid(alpha=0.3)


def panel_honesty(ax):
    # 사기: 시간분할 vs 랜덤분할 (평가 착시) + 추천 popularity
    t_pr = frd["lightgbm"]["pr_auc"]; r_pr = frr["lightgbm"]["pr_auc"]
    bars = ax.bar(["시간분할\n(정직)", "랜덤분할\n(착시)"], [t_pr, r_pr], color=[C_GOOD, C_BAD])
    annotate(ax, bars, fmt="{:.3f}", dy=0.001)
    ax.set_title("⑥ 평가 착시: 같은 데이터, 분할만 바꿔도\nPR-AUC 2.3배 부풀려짐 (사기탐지)", fontweight="bold")
    ax.set_ylabel("PR-AUC")
    ax.set_ylim(0, max(r_pr, t_pr) * 1.3)


def dashboard():
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig.suptitle("고객 재무 행동 360° — 핵심 발견 (정직한 평가가 정체성)",
                 fontsize=16, fontweight="bold", y=0.99)
    panel_leakage(axes[0, 0]); panel_ablation(axes[0, 1]); panel_interp(axes[0, 2])
    panel_income_ladder(axes[1, 0]); panel_thinfile(axes[1, 1]); panel_honesty(axes[1, 2])
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG / "00_dashboard.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def standalone(fn, name, figsize=(7, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    fn(ax); fig.tight_layout()
    fig.savefig(FIG / name, dpi=140, bbox_inches="tight"); plt.close(fig)


def journey_arc():
    """프로젝트 여정 타임라인 (1페이지 요약용)."""
    stages = [
        ("1. 숫자를 의심", "추천 평가 버그로 항상 0점, 데이터 규모 오기재(150만→1,330만)\n→ 인기도가 행렬분해를 이김"),
        ("2. 사기탐지 검증", "시간분할로 보니 약함. 랜덤분할이 PR-AUC 2.3배 부풀림\n→ 데이터 한계 인정"),
        ("3. 되는 문제 선택", "신용점수=학습불가(R²≤0), 소득=학습됨(R² 0.56)\n→ 타깃을 증명 후 선택"),
        ("4. 일반화 검증", "분포이동 테스트: 지역엔 견고, 소비수준 다르면 붕괴\n→ 어디서 깨지는지 경계선"),
        ("5. 실데이터 도입", "실제 대출 226만 건. 누수 쓰면 AUC 0.999(가짜),\n신청정보만 0.701(정직)"),
        ("6. 모델 고도화", "6기법 ablation → 이득 +0.004뿐(천장은 데이터).\n해석모델이 트리와 격차 0.01"),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 10))
    ax.axis("off")
    n = len(stages)
    ax.plot([0.12, 0.12], [0.04, 0.96], color="#1a73e8", lw=2, zorder=1)
    for i, (title, body) in enumerate(stages):
        y = 0.92 - i * (0.88 / (n - 1))
        ax.scatter(0.12, y, s=420, color="#1a73e8", zorder=2)
        ax.text(0.12, y, str(i + 1), color="white", ha="center", va="center", fontsize=12, zorder=3)
        ax.text(0.20, y + 0.012, title, fontsize=13, fontweight="bold", va="bottom", color="#1a3a6b")
        ax.text(0.20, y - 0.012, body, fontsize=10, va="top", color="#333")
    ax.set_title("프로젝트 여정 — '숫자를 의심하고 정직하게 검증한다'",
                 fontsize=14, fontweight="bold", pad=14)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.savefig(FIG / "05_journey.png", dpi=140, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    print("차트 생성 중...")
    dashboard()
    standalone(panel_leakage, "01_credit_leakage.png")
    standalone(panel_income_ladder, "03_income_ladder.png")
    standalone(panel_thinfile, "04_thinfile.png")
    journey_arc()
    print(f"저장 완료: {FIG}")
    for p in sorted(FIG.glob("*.png")):
        print("  ", p.name)
