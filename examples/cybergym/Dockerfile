FROM ghcr.io/astral-sh/uv:0.11.11 AS uv

FROM python:3.12-slim AS base

COPY --from=uv /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    UV_CACHE_DIR=/tmp/uv-cache

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

FROM base AS nooa-lock

ARG NOOA_COMMIT=8229922d7274628c9be83f745589b40852680d60
RUN git clone --filter=blob:none https://github.com/NVIDIA-NeMo/labs-OO-Agents.git /opt/nooa \
    && git -C /opt/nooa checkout --detach "$NOOA_COMMIT"
WORKDIR /opt/nooa
RUN uv export --frozen --no-dev --extra tracing --package nooa \
        --no-emit-project --no-emit-workspace \
        --output-file /tmp/nooa-requirements.txt

FROM base

ARG NOOA_COMMIT=8229922d7274628c9be83f745589b40852680d60
COPY --from=nooa-lock /tmp/nooa-requirements.txt /tmp/nooa-requirements.txt
RUN uv pip install --system --require-hashes -r /tmp/nooa-requirements.txt \
    && uv pip install --system --no-deps \
        "nooa[tracing] @ git+https://github.com/NVIDIA-NeMo/labs-OO-Agents.git@${NOOA_COMMIT}" \
    && rm /tmp/nooa-requirements.txt \
    && rm -rf /tmp/uv-cache

WORKDIR /app
RUN mkdir -p /logs/artifacts && ln -s /logs/artifacts /app/artifacts
COPY nooa_cybergym ./nooa_cybergym
COPY nooa_cybergym/llm_config.yaml ./.nooa/llm_config.yaml

CMD ["python", "-m", "nooa_cybergym.main", "--help"]
