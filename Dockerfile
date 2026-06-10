FROM python:3.13

# libgomp1 is required by PyTorch and XGBoost (OpenMP runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

COPY . .

RUN uv sync --frozen --no-dev


CMD ["uv", "run", "python"]
