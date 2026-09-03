FROM ghcr.io/astral-sh/uv:0.11.17@sha256:03bdc89bb9798628846e60c3a9ad19006c8c3c724ccd2985a33145c039a0577b AS uv

FROM python:3.12-slim-bullseye@sha256:411fa4dcfdce7e7a3057c45662beba9dcd4fa36b2e50a2bfcd6c9333e59bf0db AS build

ARG NOOA_VERSION
ARG NOOA_UV_LOCK_SHA256

COPY --from=uv /uv /uvx /bin/
RUN test -n "${NOOA_VERSION}" && test -n "${NOOA_UV_LOCK_SHA256}"

COPY source /src/labs-OO-Agents
WORKDIR /src/labs-OO-Agents
ENV UV_DYNAMIC_VERSIONING_BYPASS=${NOOA_VERSION}

RUN test "$(sha256sum uv.lock | cut -d ' ' -f 1)" = "${NOOA_UV_LOCK_SHA256}" && \
    uv export --frozen --package nooa-bench --no-dev --no-emit-workspace \
      --output-file /tmp/nooa-requirements.txt && \
    uv pip install --system --require-hashes \
      --requirements /tmp/nooa-requirements.txt && \
    uv pip install --system --no-deps --no-editable \
      . packages/nooa-cli packages/nooa-bench && \
    python -c 'import nooa, nooa_cli, nooa_bench' && \
    mkdir -p /bundle/runtime/bin /bundle/runtime/python && \
    cp -a /usr/local/. /bundle/runtime/python/ && \
    printf '%s\n' "${NOOA_VERSION}" > /bundle/runtime/VERSION && \
    find /bundle/runtime/python/lib/python3.12/lib-dynload \
      -type f -name '_tkinter*.so' -delete

COPY --chmod=0555 nemo-harbor /bundle/runtime/bin/nemo-harbor
COPY --chmod=0555 validate-installed /bundle/runtime/bin/validate-installed
COPY --chmod=0555 collect-python-libs /usr/local/bin/collect-python-libs

RUN collect-python-libs && \
    SCALED_EVALS_INSTALLED_AGENT_ROOT=/bundle/runtime \
      /bundle/runtime/bin/validate-installed

FROM debian:bullseye-slim@sha256:e5b6442dd2e9684cf5e87d8338b5968f3b348636fc0be6d7850a381e3731a2bd

COPY --from=build /bundle/runtime /opt/astra-agent/runtime
COPY copy-agent bundle.env descriptor.json source-lock.json /opt/astra-agent/

RUN chmod 0555 /opt/astra-agent/copy-agent && \
    cd /opt/astra-agent/runtime && \
    find . -type f -print0 | sort -z | xargs -0 sha256sum \
      > /opt/astra-agent/runtime.sha256

USER 1000:1000
ENTRYPOINT ["/opt/astra-agent/copy-agent"]
