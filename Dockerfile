FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1


RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*


RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /flightontime

RUN curl -LsSf https://astral.sh/uv/install.sh | sh


COPY --chown=user pyproject.toml uv.lock LICENSE ./
RUN uv sync --locked --no-default-groups --no-install-project

COPY --chown=user . .


RUN uv sync --locked --no-default-groups


ENV UV_NO_SYNC=1

EXPOSE 7860

CMD ["/flightontime/.venv/bin/uvicorn", "predicting_flight_arrival_delays.app.main:app", \
     "--host", "0.0.0.0", "--port", "7860"]
