FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CYBERGYM_TASK_DIR=/workspace \
    CYBERGYM_ARTIFACTS_DIR=/logs/artifacts \
    CYBERGYM_LOG_PATH=/logs/artifacts/log.txt \
    NEMO_OO_LLM_CONFIG=/app/nooa_cybergym/llm_config.yaml

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    binutils \
    build-essential \
    ca-certificates \
    curl \
    file \
    git \
    jq \
    less \
    ripgrep \
    strace \
    xxd \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY nooa_cybergym ./nooa_cybergym
RUN pip install .

CMD ["python", "-m", "nooa_cybergym.main", "--help"]
