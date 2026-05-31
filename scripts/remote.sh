#!/usr/bin/env bash
# Run autosub commands on a GCE VM via Docker.
# Image is built via Cloud Build and stored in Artifact Registry.
# The VM only needs Docker — no Python, uv, or ffmpeg installed directly.
# API calls stay inside GCP's network — no VPN timeouts.
#
# First-time setup:
#   remote setup       # enable APIs, create registry, create VM
#
# Day-to-day:
#   remote build       # rebuild image after code changes (runs on Cloud Build)
#   remote push <files>  # upload project files to VM
#   remote translate --profile proseka/n25 /projects/original.ass
#   remote pull <path>   # download output from VM
#   remote stop          # stop VM when done

set -euo pipefail

PROJECT="future-name-201021"
REGION="us-central1"
ZONE="${REGION}-a"
INSTANCE="autosub-runner"
MACHINE_TYPE="e2-small"
REGISTRY="${REGION}-docker.pkg.dev/${PROJECT}/autosub"
IMAGE="${REGISTRY}/autosub:latest"
GCS_BUCKET="gs://autosub-io"
SSH="gcloud compute ssh $INSTANCE --project=$PROJECT --zone=$ZONE --tunnel-through-iap --"
SCP="gcloud compute scp --project=$PROJECT --zone=$ZONE --tunnel-through-iap"

cmd="${1:-help}"
# Strip leading -- for backwards compat (--build -> build)
cmd="${cmd#--}"

case "$cmd" in
  setup)
    echo ">>> Enabling APIs..."
    gcloud services enable \
      artifactregistry.googleapis.com \
      cloudbuild.googleapis.com \
      compute.googleapis.com \
      iap.googleapis.com \
      --project="$PROJECT"

    echo ">>> Creating Artifact Registry repo..."
    gcloud artifacts repositories create autosub \
      --repository-format=docker \
      --location="$REGION" \
      --project="$PROJECT" \
      2>/dev/null || echo "    (already exists)"

    echo ">>> Creating VM..."
    gcloud compute instances create "$INSTANCE" \
      --project="$PROJECT" \
      --zone="$ZONE" \
      --machine-type="$MACHINE_TYPE" \
      --image-family=debian-12 \
      --image-project=debian-cloud \
      --boot-disk-size=30GB \
      --scopes=cloud-platform

    echo ">>> Waiting for boot (~30s)..."
    sleep 30

    echo ">>> Installing Docker + configuring registry auth..."
    $SSH bash -s <<EOF
set -euo pipefail
sudo apt-get update -qq
sudo apt-get install -y -qq docker.io > /dev/null 2>&1
sudo gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet
mkdir -p ~/projects
sudo docker version
echo "=== Docker installed ==="
EOF

    echo ""
    echo "=== Setup complete ==="
    echo "Next: bash scripts/remote.sh build"
    exit 0
    ;;

  build)
    echo ">>> Building image via Cloud Build..."
    SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
    gcloud builds submit "$SCRIPT_DIR" \
      --tag="$IMAGE" \
      --project="$PROJECT"
    echo "=== Image pushed to $IMAGE ==="
    echo "The VM will pull it automatically on next run."
    exit 0
    ;;

  start)
    echo ">>> Starting VM..."
    gcloud compute instances start "$INSTANCE" --project="$PROJECT" --zone="$ZONE"
    exit 0
    ;;

  stop)
    echo ">>> Stopping VM (disk preserved, no compute charges)..."
    gcloud compute instances stop "$INSTANCE" --project="$PROJECT" --zone="$ZONE"
    exit 0
    ;;

  ssh)
    gcloud compute ssh "$INSTANCE" --project="$PROJECT" --zone="$ZONE" --tunnel-through-iap
    exit 0
    ;;

  push)
    shift
    for f in "$@"; do
      BASENAME="$(basename "$f")"
      echo ">>> Uploading $f to GCS..."
      gcloud storage cp "$f" "${GCS_BUCKET}/${BASENAME}"
      echo ">>> Fetching to VM (internal network)..."
      $SSH gcloud storage cp "${GCS_BUCKET}/${BASENAME}" ~/projects/
    done
    echo "=== Files available at /projects/ inside the container ==="
    exit 0
    ;;

  pull)
    shift
    REMOTE_PATH="${1:?Usage: remote pull <path-on-vm>}"
    echo ">>> Downloading $REMOTE_PATH"
    $SCP "$INSTANCE:$REMOTE_PATH" .
    exit 0
    ;;

  help)
    echo "Usage: remote <command> [args...]"
    echo ""
    echo "Commands:"
    echo "  setup              First-time: enable APIs, create registry + VM"
    echo "  build              Build image via Cloud Build (after code changes)"
    echo "  push <files...>    Upload project files to VM"
    echo "  pull <path>        Download files from VM"
    echo "  start              Start a stopped VM"
    echo "  stop               Stop VM (preserves disk, no compute cost)"
    echo "  ssh                Interactive SSH session"
    echo "  <autosub args>     Run autosub command on the VM"
    echo ""
    echo "Examples:"
    echo "  remote build"
    echo "  remote push projects/projects/N25/ep01/original.ass"
    echo "  remote translate --profile proseka/n25 /projects/original.ass"
    echo "  remote pull ~/projects/translated.ass"
    exit 0
    ;;
esac

# Default: run autosub via Docker on the VM.
# Pulls latest image, mounts ~/projects at /projects.
# Each argument is shell-quoted so spaces in filenames survive the SSH hop.
echo ">>> autosub $*"

QUOTED_ARGS=""
for arg in "$@"; do
  QUOTED_ARGS="$QUOTED_ARGS '$(echo "$arg" | sed "s/'/'\\\\''/g")'"
done

$SSH "sudo docker pull $IMAGE --quiet && sudo docker run --rm \
  -v \$HOME/projects:/projects \
  -e GOOGLE_CLOUD_PROJECT=$PROJECT \
  -e AUTOSUB_GCS_BUCKET=subtitling-projects \
  $IMAGE $QUOTED_ARGS"
