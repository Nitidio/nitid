# nitid Web Application

A browser-based interface for running object detection with nitid. Upload images or videos, choose a model, tune parameters, and browse annotated results with per-user persistent history.

---

## Architecture

```
web/
├── api/           FastAPI REST backend
└── frontend/      React 18 + Vite SPA (TypeScript)

models/            Pre-placed .pth checkpoints (git-ignored)
web/storage/       Runtime data: uploads + result snapshots (git-ignored)
```

The frontend and backend are fully decoupled. The React SPA talks to the API over HTTP; both can run on the same machine during development.

---

## Installation

Install the web extras alongside the base package:

```bash
uv sync --extra web
```

This adds: `fastapi`, `uvicorn`, `sqlalchemy`, `bcrypt`, `python-jose`, `pydantic-settings`, `python-multipart`.

The frontend requires Node.js 18+:

```bash
cd web/frontend
npm install
```

---

## Running

Both processes must be started from the **project root**:

```bash
# 1 — API server (keep to a single worker — model inference is not thread-safe)
uv run uvicorn web.api.main:app --workers 1

# 2 — Frontend dev server (separate terminal)
cd web/frontend && npm run dev
# → http://localhost:5173
```

On first start the API creates `web/storage/nitid.db` with all tables and the `web/storage/uploads/` and `web/storage/results/` directories automatically.

### Place model checkpoints

Put nitid `.pth` checkpoints in `models/` at the project root. The API lists
everything it finds there:

```bash
mkdir -p models
uv run dfine download model=dfine_l output=models
```

### Environment variables

All settings use the `NITID_` prefix and can be overridden in a `.env` file at the project root:

| Variable | Default | Description |
|---|---|---|
| `NITID_SECRET_KEY` | `change-me-in-production` | JWT signing secret — change before exposing to a network |
| `NITID_DATABASE_URL` | `sqlite:///./web/storage/nitid.db` | SQLAlchemy URL |
| `NITID_MODELS_DIR` | `./models` | Directory scanned for `.pth` checkpoints |
| `NITID_UPLOADS_DIR` | `./web/storage/uploads` | Uploaded source files |
| `NITID_RESULTS_DIR` | `./web/storage/results` | Annotated result snapshots |
| `NITID_ACCESS_TOKEN_EXPIRE_MINUTES` | `480` | JWT lifetime (8 hours) |

---

## Using the web UI

### 1. Create an account

Navigate to `http://localhost:5173`. Click **Register**, enter a username (≥ 3 chars) and password (≥ 8 chars).

### 2. Start a new run

Click **New Run** and follow the three-step form:

1. **Select model** — pick from checkpoints in `models/`. The class list is loaded automatically.
2. **Configure parameters**:
   - *Confidence threshold* — slider from 0.01 to 1.0 (default 0.5)
   - *Image size* — 320 / 480 / 640 / 960 / 1280 (default 640; must match `eval_spatial_size` in the checkpoint)
   - *Filter classes* — optional comma-separated class IDs; empty means detect all
   - *Frame step* — for video inputs only; sample 1 frame every N frames (default 30)
3. **Upload files** — drag-and-drop or click to browse. Accepts images (`.jpg .jpeg .png .bmp .tiff .webp`) and videos (`.mp4 .avi .mov .mkv .ts .m4v`). Mixing images and a video in the same upload is not supported; if any file is a video the whole run is treated as a video run.

Click **Run detection**. The page navigates to the run detail view immediately.

### 3. View results

The run detail page shows a grid of annotated snapshots. Each card displays:
- The image with bounding boxes drawn by the model
- Number of detections
- Frame index (video runs only)

Click any card to open the detail panel: full-size annotated image on the left, detection table on the right (class name, confidence %, and pixel coordinates).

The page auto-refreshes every 2 seconds until the run reaches `done` or `failed`.

### 4. Run history

The **Runs** list shows all past runs for the logged-in user with status, model, input type, item count, and creation time. Click **View** to revisit any run, or **Delete** to remove it along with all uploaded files and result snapshots.

---

## REST API reference

The API is self-documented at `http://localhost:8000/docs` (Swagger UI).

### Authentication

All endpoints except `/auth/register` and `/auth/login` require a JWT in the `Authorization: Bearer <token>` header.

| Method | Path | Description |
|---|---|---|
| `POST` | `/auth/register` | Create account. Body: `{"username": "…", "password": "…"}` |
| `POST` | `/auth/login` | Login (`application/x-www-form-urlencoded`). Returns `{"access_token": "…"}` |
| `GET` | `/auth/me` | Current user info |

### Models

| Method | Path | Description |
|---|---|---|
| `GET` | `/models` | List available `.pth` filenames in `models/` |
| `GET` | `/models/{name}/classes` | Class ID → name mapping for a specific checkpoint |

### Runs

| Method | Path | Description |
|---|---|---|
| `POST` | `/runs` | Create and enqueue a run. Returns `202` immediately. |
| `GET` | `/runs` | List the current user's runs (newest first). Query params: `skip`, `limit` |
| `GET` | `/runs/{id}` | Full run detail including all items and detections |
| `DELETE` | `/runs/{id}` | Delete run, items, and stored files |

**Creating a run** (`POST /runs`) uses `multipart/form-data`:
- `files` — one or more uploaded files
- `params` — JSON-encoded run parameters:

```json
{
  "model_name": "dfine_l_wrapped.pth",
  "conf": 0.5,
  "imgsz": 640,
  "classes": null,
  "frame_step": 30
}
```

**Run status** lifecycle: `pending` → `running` → `done` | `failed`. Poll `GET /runs/{id}` while status is `pending` or `running`.

### Files

| Method | Path | Description |
|---|---|---|
| `GET` | `/files/{run_id}/{filename}?token=<jwt>` | Serve a result snapshot JPEG. Token is passed as a query parameter so `<img src>` tags can load images directly. |

---

## Data model

### `runs` table

| Column | Type | Notes |
|---|---|---|
| `id` | int PK | |
| `user_id` | int FK | → `users.id` |
| `model_name` | str | Checkpoint filename |
| `conf` | float | Confidence threshold used |
| `imgsz` | int | Input image size |
| `classes` | str (JSON) | `[0,1,2]` or `null` |
| `frame_step` | int | Video sampling interval |
| `status` | str | `pending` / `running` / `done` / `failed` |
| `input_type` | str | `images` or `video` |
| `created_at` | datetime | |
| `completed_at` | datetime | Set when done or failed |
| `error_msg` | text | Python traceback if failed |

### `run_items` table

One row per processed image or sampled video frame.

| Column | Type | Notes |
|---|---|---|
| `id` | int PK | |
| `run_id` | int FK | → `runs.id` (cascade delete) |
| `source_path` | text | Absolute path to the uploaded file |
| `result_snapshot_path` | text | Filename (within `results/{run_id}/`) of the annotated JPEG |
| `detections_json` | text | JSON array of detections from `result.to_json()` |
| `frame_idx` | int | Video frame number; `null` for image runs |

---

## Implementation notes

### Single worker requirement

The model is not thread-safe for concurrent inference. Always start uvicorn with `--workers 1`. The in-process model cache (`_model_cache` in `services/inference.py`) is process-local; running multiple workers would create separate caches and double memory usage.

### Model cache

`get_model(model_name)` in `services/inference.py` returns a cached `DFINE` instance keyed by absolute checkpoint path. The first call per model loads weights (~100–200 MB) and fuses BatchNorm layers (one-way, but idempotent). All subsequent calls for the same model return the already-deployed instance with no overhead.

### Background inference

`POST /runs` saves the uploaded files, creates the `Run` row with `status=pending`, and returns `202` before inference starts. Inference runs as a FastAPI `BackgroundTask` using its own `SessionLocal()` database session (not the request session, which closes when the handler returns).

### Video sampling

Videos are not decoded through `LoadSource`. Instead `cv2.VideoCapture` iterates frames directly, skipping `frame_step - 1` frames between samples. For a 30 fps video with the default `frame_step=30`, one frame per second is processed.

### Annotated images

Result images are generated server-side via `result.plot()` (the same method available in the Python API) and written as JPEGs under `web/storage/results/{run_id}/`. The frontend loads them via the `/files/` endpoint, which validates that the requesting user owns the run.
