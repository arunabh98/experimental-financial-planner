FROM python:3.11-slim

RUN useradd -m -u 1000 user

WORKDIR /app

COPY --chown=user poetry.lock pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

RUN mkdir -p /app/coding

RUN chown user:user /app/coding

USER user

CMD ["uvicorn", "financial_planner.app:app", "--host", "0.0.0.0", "--port", "8000"]
