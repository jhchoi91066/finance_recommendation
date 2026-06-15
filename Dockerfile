# 신용위험 스코어링 API — 컨테이너 이미지
# 빌드: docker build -t credit-scoring-api .
# 실행: docker run -p 8000:8000 credit-scoring-api   → http://localhost:8000/docs
#
# 주의: 먼저 호스트에서 `python src/train_model.py` 로 models/ 를 생성해야 한다.
#       (1.2GB 학습 데이터는 이미지에 넣지 않는다 — 가벼운 모델 아티팩트만 싣는다.)
FROM python:3.11-slim

# LightGBM 런타임 의존성(OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# 서빙에 필요한 것만 복사 (학습 코드·대용량 데이터 제외)
COPY src/api.py src/api.py
COPY models/ models/

EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
