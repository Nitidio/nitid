# macOS Developer Setup (via Docker)

> **Who this is for:** Developers on Apple Silicon (M-series) Macs.  
> The project's default dependencies include CUDA and NVIDIA libraries that are incompatible with macOS natively. This guide runs the project inside a Linux container on your Mac so your environment is identical to your Linux teammates — no changes to `pyproject.toml` needed.

---

## Prerequisites

### 1. Install Docker Desktop for Mac
Download and install from the official site:  
**[https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)**

- Choose the **Apple Silicon** version during download.
- Once installed, open Docker Desktop and wait until the status indicator in the menu bar shows **"Engine running"** (green icon).

> **Disk space:** The container downloads ~3.5 GB of CUDA/PyTorch packages to your Mac filesystem. Make sure you have at least **8 GB free** before continuing.

---

## Why this approach works

Normally, `uv sync` on macOS would fail because CUDA packages target Linux x86_64. Running inside a `linux/amd64` Docker container gives us a real Linux environment. The key design decision is:

- **Python packages are NOT baked into the Docker image.** They are installed into a `.venv` folder inside your project directory — which lives on your Mac's own filesystem via a volume mount. This avoids Docker's internal virtual disk, which is too small to hold 3.5 GB of CUDA libraries.
- The `Dockerfile` and `docker-compose.yml` are **local-only** files and do not affect the shared repo.

---

## Setup Files

These two files must exist in your project root. They are already present in the repo — do not modify them.

### `Dockerfile`
```dockerfile
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
```

### `docker-compose.yml`
```yaml
services:
  nitid-dev:
    image: nitid-env
    container_name: nitid_container

    volumes:
      # Bind-mounts your Mac project folder into /app inside the container.
      # .venv will be created HERE (on your Mac disk), bypassing Docker's virtual disk.
      - .:/app

    environment:
      - PYTHONUNBUFFERED=1
      - PYTHONDONTWRITEBYTECODE=1
      - UV_LINK_MODE=copy

    # Keep the container alive with an interactive shell.
    stdin_open: true
    tty: true
    restart: unless-stopped
```

---

## One-time Setup

Open your Mac terminal, navigate to the project root, and run the following commands **in order**.

### Step 1 — Clone the repo
```bash
git clone <repo> && cd nitid
```

### Step 2 — Build the Docker image (~1–2 minutes)
This builds a lightweight Linux image containing only system tools and `uv`. No Python packages are downloaded at this stage.
```bash
docker build -t nitid-env .
```

You should see it finish with:
```
Successfully tagged nitid-env:latest
```

### Step 3 — Start the container
```bash
docker compose up -d
```

### Step 4 — Enter the Linux terminal
```bash
docker compose exec -it nitid-dev bash
```

Your prompt will change to something like `root@<id>:/app#` — you are now inside a Linux shell. Your entire Mac project folder is live-mounted at `/app`.

### Step 5 — Install Python packages (run once inside the container, takes ~5–10 min)
```bash
uv sync --frozen --extra dev
```

This downloads all packages (PyTorch, CUDA libraries, pytest, ruff, mypy, etc.) directly into `.venv/` on your Mac filesystem. You will see a live progress bar.

### Step 6 — Verify the setup (from `onboarding.md`)
```bash
uv run pytest tests/unit           # should show: 42 passed
uv run ruff check .
uv run mypy dfine/
```

Scope Ruff and mypy to the maintained package, tools, tests, and scripts as shown above.

---

## Daily workflow

Every time you start a new work session:

```bash
# 1. Make sure Docker Desktop is running (check the menu bar icon)

# 2. Start your container (if not already running)
docker compose up -d

# 3. Enter the Linux shell
docker compose exec -it nitid-dev bash

# 4. You're in — start working!
```

> **The `.venv` is persistent.** Since it lives in your project folder on your Mac, you do not need to re-run `uv sync` every session — only when `uv.lock` changes (i.e. after a `git pull` that updates dependencies).

---

## Stopping the container

```bash
# Pause it (keeps .venv intact, fast to restart)
docker compose stop

# Or remove it entirely (safe — .venv on your Mac is untouched)
docker compose down
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `docker compose up` gives `500 Internal Server Error` | Docker virtual disk corrupted | Open Docker Desktop → Troubleshoot → Clean/Purge data, then rebuild with `docker build -t nitid-env .` |
| `No space left on device` during `uv sync` | Mac disk is full | Free up space, delete `.venv/`, then re-run `uv sync --frozen --extra dev` |
| `torch.cuda.is_available()` returns `False` | No NVIDIA GPU on Mac | Expected — you can write code and run unit tests without a GPU |
| Container shows a `linux/amd64` platform warning | Running via Rosetta 2 on Apple Silicon | Harmless — the container works correctly |
| `uv sync` fails with `Unable to find lockfile` | `uv.lock` was not copied into the build context | Make sure `uv.lock` exists in the project root before running `docker build` |
