#!/usr/bin/env python3
"""
신용위험 스코어링 API (FastAPI 서빙)
====================================

오프라인 학습된 모델(`models/`)을 로드해 대출 신청을 실시간 평가한다.
대출 신청 JSON → 연체확률 + 신용점수 + 의사결정 + 사유코드(SHAP).

실행
  uvicorn src.api:app --reload          # 개발
  uvicorn src.api:app --host 0.0.0.0 --port 8000   # 운영
  → Swagger UI: http://localhost:8000/docs

학습이 선행돼야 한다: python src/train_model.py  (models/ 생성)
"""

import json
import math
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
# 저장소 실행(src/api.py)·컨테이너 실행 모두에서 models/ 를 찾도록 견고화
_CANDIDATES = [ROOT / "models", Path(__file__).resolve().parent / "models", Path("models")]
MODELS = next((p for p in _CANDIDATES if (p / "credit_model.txt").exists()), ROOT / "models")

app = FastAPI(
    title="신용위험 스코어링 API",
    description="대출 신청을 받아 연체확률·신용점수·의사결정·사유코드를 반환하는 여신 심사 서비스",
    version="1.0.0",
)

# ---- 모델·메타 로드 (서버 시작 시 1회) ----
_booster = None
_meta = None


def _load():
    global _booster, _meta
    if _booster is None:
        import lightgbm as lgb
        _booster = lgb.Booster(model_file=str(MODELS / "credit_model.txt"))
        _meta = json.loads((MODELS / "model_meta.json").read_text())
    return _booster, _meta


# ---- 입력 스키마 ----
class LoanApplication(BaseModel):
    loan_amnt: float = Field(..., description="대출 신청 금액($)", examples=[15000])
    annual_inc: float = Field(..., description="연소득($)", examples=[45000])
    dti: float = Field(..., description="부채/소득비(DTI, %)", examples=[18.5])
    term: int = Field(36, description="대출 기간(개월): 36 또는 60", examples=[36])
    emp_length: int = Field(5, description="근속연수(0~10, 10은 10년+)", examples=[5])
    home_ownership: str = Field("MORTGAGE", description="주거: RENT/MORTGAGE/OWN 등", examples=["MORTGAGE"])
    purpose: str = Field("debt_consolidation", description="대출 목적", examples=["debt_consolidation"])
    installment: float = Field(..., description="월 상환액($)", examples=[480])
    open_acc: int = Field(10, description="활성 신용 계좌 수", examples=[10])
    revol_util: float = Field(45.0, description="신용 회전율(%)", examples=[45.0])
    total_acc: int = Field(25, description="총 신용 계좌 수", examples=[25])


def _build_vector(app_in: LoanApplication, meta):
    inc = max(app_in.annual_inc, 1)
    ho = meta["cat_maps"]["home_ownership"]
    pp = meta["cat_maps"]["purpose"]
    feats = {
        "loan_amnt": app_in.loan_amnt, "annual_inc": app_in.annual_inc, "dti": app_in.dti,
        "open_acc": app_in.open_acc, "revol_util": app_in.revol_util, "total_acc": app_in.total_acc,
        "installment": app_in.installment, "term_months": app_in.term, "emp_years": app_in.emp_length,
        "loan_to_income": app_in.loan_amnt / inc,
        "pay_to_income": (app_in.installment * 12) / inc,
        "inst_to_loan": app_in.installment / (app_in.loan_amnt + 1),
        "home_ownership_code": ho.index(app_in.home_ownership) if app_in.home_ownership in ho else -1,
        "purpose_code": pp.index(app_in.purpose) if app_in.purpose in pp else -1,
    }
    vec = np.array([[feats[f] for f in meta["feature_order"]]], dtype=float)
    return vec, feats


def _score(p, sp):
    factor = sp["pdo"] / math.log(2)
    offset = sp["base"] - factor * math.log(sp["base_odds"])
    odds_good = (1 - p) / max(p, 1e-6)
    return int(np.clip(offset + factor * math.log(max(odds_good, 1e-6)), 300, 850))


def _reasons(vec, feats, meta, k=4):
    booster, _ = _load()
    contrib = booster.predict(vec, pred_contrib=True)[0]   # (F+1,), 마지막=base
    shap = contrib[:-1]
    order = np.argsort(np.abs(shap))[::-1][:k]
    labels = meta["feature_labels"]
    cat_rev = {"home_ownership_code": meta["cat_maps"]["home_ownership"],
               "purpose_code": meta["cat_maps"]["purpose"]}
    out = []
    for i in order:
        fname = meta["feature_order"][i]
        raw = feats[fname]
        # 범주형은 코드 대신 실제 값(MORTGAGE 등)으로 표기
        if fname in cat_rev:
            idx = int(raw)
            value = cat_rev[fname][idx] if 0 <= idx < len(cat_rev[fname]) else "기타"
        else:
            value = round(float(raw), 2)
        out.append({
            "factor": labels.get(fname, fname),
            "value": value,
            "effect": "위험↑" if shap[i] > 0 else "위험↓",
            "impact": round(float(shap[i]), 3),
        })
    return out


@app.get("/health")
def health():
    _, meta = _load()
    return {"status": "ok", "model": "credit_default_lgbm",
            "validation_auc": meta["validation_auc"], "n_train": meta["n_train"]}


@app.post("/score")
def score(application: LoanApplication):
    booster, meta = _load()
    vec, feats = _build_vector(application, meta)
    p = float(booster.predict(vec)[0])
    th = meta["thresholds"]
    decision = ("승인" if p < th["approve_below"]
                else "거절" if p > th["reject_above"] else "검토")
    return {
        "default_probability": round(p, 4),
        "credit_score": _score(p, meta["score"]),
        "decision": decision,
        "reason_codes": _reasons(vec, feats, meta),
        "model_validation_auc": meta["validation_auc"],
    }


@app.get("/")
def root():
    return {"service": "신용위험 스코어링 API", "docs": "/docs", "score": "POST /score"}
