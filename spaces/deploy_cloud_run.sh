#!/usr/bin/env bash
# Deploy the demo in spaces/demo/ to Google Cloud Run.
#
#   PROJECT=my-gcp-project ./spaces/deploy_cloud_run.sh
#
# Environment:
#   PROJECT   Google Cloud project id (required)
#   REGION    default europe-west1 (a tier-1 region, so the free quota goes further)
#   SERVICE   default nitid-demo
#   CPU       default 2
#   MEMORY    default 4Gi
#   DRY_RUN=1 print the command and do nothing
#
# The cost limits (--min-instances 0, --max-instances 1) are fixed on purpose: a Google Cloud
# budget only sends alerts, it does not stop spending. See spaces/README.md.
set -euo pipefail

: "${PROJECT:?Set PROJECT to your Google Cloud project id}"
REGION="${REGION:-europe-west1}"
SERVICE="${SERVICE:-nitid-demo}"
CPU="${CPU:-2}"
MEMORY="${MEMORY:-4Gi}"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cmd=(
  gcloud run deploy "$SERVICE"
  --project "$PROJECT"
  --source "$here/demo"
  --region "$REGION"
  --allow-unauthenticated
  --port 8080
  --cpu "$CPU" --memory "$MEMORY"
  --cpu-boost
  --min-instances 0 --max-instances 1
  --timeout 300
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

command -v gcloud >/dev/null || { echo "gcloud is not installed" >&2; exit 1; }
"${cmd[@]}"
