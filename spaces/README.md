# Public demo

`demo/` is the source of the interactive demo ([#7](https://github.com/Nitidio/nitid/issues/7)).
It is a small Gradio app that runs the **published** `nitid` package from PyPI on CPU, so what
visitors try is what `pip install nitid` gives them. It is built as a container image and runs on
Google Cloud Run.

Hugging Face no longer lets free accounts create Gradio or Docker Spaces, so the demo does not
target one. `demo/README.md` still carries a Space card header; it is ignored by Cloud Run and
lets the same folder be uploaded to a Space if someone with a paid plan wants a mirror.

| File | Purpose |
|---|---|
| `demo/app.py` | The app. Plain functions plus `build_demo()`; the only `gradio` import is inside it. |
| `demo/requirements.txt` | Installs `nitid` (and CPU PyTorch wheels). |
| `demo/Dockerfile` | Container image: Python 3.12, Gradio 6.32.0, weights downloaded at build time. |
| `demo/.dockerignore` | Keeps caches and the Space card out of the image. |
| `demo/examples/` | Example images offered in the interface. |
| `deploy_cloud_run.sh` | `gcloud run deploy` with fixed cost limits. |

## Run it locally

```bash
uv sync --extra dev
uv pip install gradio==6.32.0   # not a dependency of nitid
uv run python spaces/demo/app.py
```

The container gets its host and port from `GRADIO_SERVER_NAME` and `GRADIO_SERVER_PORT`; the app itself
does not read them.

To try the image locally (needs Docker and access to download.pytorch.org, GitHub and huggingface.co):

```bash
docker build -t nitid-demo spaces/demo
docker run --rm -p 8080:8080 nitid-demo      # then open http://localhost:8080
```

## Deploy to Google Cloud Run

Cloud Run bills per request, scales to zero when nobody visits, and has a monthly free allowance per
billing account (2 million requests, 180,000 vCPU-seconds, 360,000 GiB-seconds at the time of writing;
check <https://cloud.google.com/run/pricing>). It needs a billing account with a card.

One-time setup:

1. `gcloud auth login`
2. Create a project and link billing:
   ```bash
   gcloud projects create <project> --name="nitid demo"
   gcloud billing accounts list
   gcloud billing projects link <project> --billing-account=<account>
   ```
3. Create a budget with alerts in the console (Billing > Budgets & alerts). A budget **only sends
   alerts; it does not stop spending**. The real brake is the instance limit set by the script.
4. Enable the APIs:
   ```bash
   gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
   ```

Deploy (from the repository root):

```bash
PROJECT=<project> ./spaces/deploy_cloud_run.sh          # DRY_RUN=1 prints the command only
```

The script uses region `europe-west1` (override with `REGION`), 2 vCPU and 4 GiB (override with `CPU`
and `MEMORY`), and always sets `--min-instances 0 --max-instances 1`. It builds the image with Cloud
Build from `demo/Dockerfile`, stores it in Artifact Registry and prints the service URL.

Check it:

```bash
gcloud run services describe nitid-demo --region europe-west1 --format='value(status.url)'
gcloud run services logs read nitid-demo --region europe-west1 --limit 50
```

The first visit after a quiet period is a cold start (PyTorch has to load), so expect a wait. The logs
must **not** show weights being downloaded: they are part of the image. After a few days, check
Billing > Reports to confirm usage stays inside the free allowance, and lower `CPU`/`MEMORY` if the
service metrics allow it.

To update the demo, change `spaces/demo/` and run the script again. Each deploy stores a new image,
so set a cleanup policy on the `cloud-run-source-deploy` Artifact Registry repository. To remove the
demo: `gcloud run services delete nitid-demo --region europe-west1`.

## Design notes

- `nitid>=0.9.4` with a fallback: releases up to 0.9.4 only know the `model1<size>` names, so
  `load_model` tries `nitid1<size>` first and `model1<size>` second. Once a release with the new
  names is on PyPI, the requirement can be raised and the fallback removed.
- Weights are fetched while the image builds (`detect`/`s`, `segment`/`s`, `segment`/`n`). Cloud Run's
  filesystem lives in memory, and `nitid` reuses a checkpoint already present in the working
  directory (`/app`), so nothing is downloaded when the container starts. Sizes `m`, `l` and `x` are
  downloaded on first use, into memory.
- There are no official nano **detection** weights, so size `n` is only offered for segmentation.
- At most two models stay in memory (`lru_cache(maxsize=2)`); a CPU container has little RAM.
- Tests: `tests/unit/test_space_demo.py` (logic, fake model), `tests/unit/test_space_deploy.py`
  (Dockerfile and script stay consistent: port, Gradio version, prefetched sizes, cost limits),
  `tests/integration/test_space_demo.py` (real tiny models) and `tests/integration/test_space_server.py`
  (the app serves on the port given by the environment). None builds the image or contacts Google Cloud.
- Not verified yet: the image build and a real deployment (they need network access to PyTorch's wheel
  index, GitHub and huggingface.co, and a Google Cloud project). The first deployment is the final test.
