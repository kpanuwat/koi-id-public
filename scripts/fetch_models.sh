#!/usr/bin/env bash
# Download the model parts the app needs (git holds only app/models/parts.json) from this repo's GitHub Release.
#   scripts/fetch_models.sh [TAG]        default TAG = models-v0.064
# Needs curl. Parts are <= 20 MB each (Cloudflare Pages file limit); app/models.js joins them at load time.
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=${1:-models-v0.064}
REPO=${KOI_REPO:-kpanuwat/koi-id-public}   # set to the public repo, e.g. KOI_REPO=yourname/koi-id
BASE="https://github.com/$REPO/releases/download/$TAG"
for model in koi_yx320.onnx embed.onnx; do
  n=$(python3 -c "import json,sys; print(json.load(open('app/models/parts.json'))['$model']['parts'])")
  for ((i = 0; i < n; i++)); do
    f="app/models/$model.part$i"
    [ -s "$f" ] && continue
    echo "downloading $f"; curl -fL --retry 3 -o "$f" "$BASE/$model.part$i"
  done
done
echo "models ready in app/models/"
