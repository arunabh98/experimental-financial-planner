FROM python:3.11-slim

RUN useradd -m -u 1000 user

WORKDIR /app

COPY --chown=user poetry.lock pyproject.toml ./
COPY --chown=user requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=user . /app

USER user

CMD ["uvicorn", "financial_planner.app:app", "--host", "0.0.0.0", "--port", "7860"]
