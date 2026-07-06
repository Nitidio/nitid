# ── Nitid Developer Environment ─────────────────────────────────────────────
# Strategy: Only bake system tools into the image.
# Python packages are installed at container startup via "uv sync", which
# writes into the .venv folder on the bind-mounted Mac filesystem (/app).
# This completely bypasses Docker's virtual disk overlay — no more I/O errors.
FROM --platform=linux/amd64 python:3.12-slim

# Install system libraries needed by the project (OpenCV, git, curl, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    glib2.0-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv — the fast Python package manager used by this project
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /root/.local/bin/uvx /bin/

# Set working directory — this is where the Mac folder will be mounted
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy

# Default: open an interactive bash shell
CMD ["/bin/bash"]
