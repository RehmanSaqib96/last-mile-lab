# Small image that serves the API.  Build:  docker build -t last-mile-lab .
#                                    Run:    docker run -p 8000:8000 last-mile-lab
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "lastmile.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
