# QueryMesh application image.
#
# Multi-stage: a `builder` stage installs dependencies into a venv, then
# the `runtime` stage copies just the venv + source, so the final image
# doesn't carry build toolchains. Does NOT include the `docling` extra
# (heavy ML deps — torch/transformers) or Airflow (real dependency
# conflicts with this stack, see querymesh/airflow/requirements.txt) —
# both are separate, deliberate exclusions, not oversights.

FROM python:3.12-slim AS builder

WORKDIR /build

# Only copy dependency-defining files first, so Docker's layer cache is
# reused across builds that only change application code.
COPY pyproject.toml README.md requirements-lock.txt ./
COPY querymesh ./querymesh

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-lock.txt \
    && pip install --no-cache-dir --no-deps .


FROM python:3.12-slim AS runtime

# Non-root user — never run the app as root in a container.
RUN groupadd --gid 1000 querymesh \
    && useradd --uid 1000 --gid querymesh --shell /bin/bash --create-home querymesh

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY --chown=querymesh:querymesh querymesh ./querymesh
COPY --chown=querymesh:querymesh pyproject.toml README.md ./

USER querymesh

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -m querymesh health || exit 1

# Default: launch the Gradio UI. Override with `docker run ... python -m querymesh health`
# or any other entrypoint for one-off commands (e.g. ingestion scripts).
EXPOSE 7860
CMD ["python", "-c", "from querymesh.ui import main; main()"]
