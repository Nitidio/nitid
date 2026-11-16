# Container image for the public nitid demo on Google Cloud Run.
# See docs: spaces/README.md (build, run and deploy instructions).
FROM python:3.12-slim

# Cloud Run sends requests to $PORT (8080 by default) on 0.0.0.0. Gradio reads these two variables.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=8080

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# requirements.txt carries the CPU-only PyTorch wheel index. Gradio is pinned to the version
# the demo was developed and tested with.
COPY requirements.txt .
RUN pip install -r requirements.txt "gradio==6.32.0"

COPY . .

# Download the official weights while building. Cloud Run's filesystem lives in memory, so weights
# fetched at runtime would use RAM and slow every cold start. nitid reuses a checkpoint that already
# exists in the working directory (/app), so nothing is downloaded when the container starts.
# Sizes m, l and x are downloaded on first use instead, to keep the image small.
RUN python -c "import app; [app.load_model(t, s) for t, s in (('detect', 's'), ('segment', 's'), ('segment', 'n'))]"

CMD ["python", "app.py"]
