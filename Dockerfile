FROM python:3.11.11-slim

RUN groupadd -g 1000 usergroup && \
    useradd -m -u 1000 -g usergroup user

WORKDIR /app

COPY --chown=user:usergroup poetry.lock pyproject.toml ./

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev --no-interaction --no-ansi

COPY --chown=user:usergroup . .

RUN mkdir -p /app/coding && \
    chown user:usergroup /app/coding && \
    chmod 755 /app/coding

USER user

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

CMD ["uvicorn", "financial_planner.app:app", "--host", "0.0.0.0", "--port", "8000"]
