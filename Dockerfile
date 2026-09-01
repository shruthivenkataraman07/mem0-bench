FROM python:3.11-slim AS base

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
COPY configs ./configs
COPY experiments ./experiments
COPY dashboard ./dashboard
COPY README.md ./

RUN pip install --no-cache-dir -e ".[dashboard]"

COPY datasets ./datasets

ENTRYPOINT ["mem0-bench"]
CMD ["--help"]
