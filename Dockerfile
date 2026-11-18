# Hybrid RAG Assistant: one container, CPU only.
#
#   docker build -t hybrid-rag .
#   docker run -p 8000:8000 --env-file .env -v rag-data:/data hybrid-rag
#
# The Groq key comes from --env-file at run time and is never baked into the
# image. Accounts, sessions and every user's PDFs and index live in the /data
# volume, so they survive image upgrades. Serve it behind HTTPS (a reverse
# proxy) and set APP_SECURE_COOKIES=1 before exposing it beyond localhost.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    APP_DB_PATH=/data/app.db \
    APP_STORAGE_DIR=/data/storage

WORKDIR /app

# CPU-only PyTorch first: the default wheel bundles CUDA and is several GB larger.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding model into the image so containers start without a download.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
ENV HF_HUB_OFFLINE=1

COPY . .

# Run as an unprivileged user that can write only to /data.
RUN useradd --create-home --uid 10001 app && mkdir -p /data && chown app /data
USER app
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4)"

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
