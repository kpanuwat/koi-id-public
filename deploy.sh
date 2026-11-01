#!/usr/bin/env bash
# Publish app/ to Cloudflare Pages (HTTPS, needed for the phone camera). Ships no pond videos, no dev pages and
# no whole .onnx files (models go as <=20 MB parts, see train/split_models.py). Needs `npx wrangler login` once.
set -e
cd "$(cd "$(dirname "$0")" && pwd)"
[ -s ~/.nvm/nvm.sh ] && source ~/.nvm/nvm.sh >/dev/null || true
# Cloudflare API token for wrangler when set (Linux VM); the Mac uses wrangler OAuth instead
[ -f ~/.config/koi-id/cf.env ] && { set -a; . ~/.config/koi-id/cf.env; set +a; }
# models + onnxruntime wasm: sw.js caches them under a name made from their bytes, so code-only releases keep them on
# phones; after a model change the name must change too (stops once: commit the updated sw.js and run again)
MH=$(cat app/models/*.onnx.part* app/vendor/*.wasm | sha1sum | cut -c1-12)
if ! grep -q "const MODELS = 'koi-models-$MH';" app/sw.js; then
  perl -pi -e "s/const MODELS = 'koi-models-[0-9a-z]*';/const MODELS = 'koi-models-$MH';/" app/sw.js
  echo "models changed: app/sw.js MODELS is now koi-models-$MH; commit it, then run ./deploy.sh again"; exit 1
fi
rsync -a --delete --exclude 'models/*.mp4' --exclude 'models/*.onnx' --exclude dev --exclude '*.test.mjs' app/ dist/
# cross-origin isolation from the first load (wasm threads); the service worker adds the same headers offline
printf '/*\n  Cross-Origin-Opener-Policy: same-origin\n  Cross-Origin-Embedder-Policy: require-corp\n' > dist/_headers
npx --yes wrangler@4 pages deploy dist --project-name "${KOI_PROJECT:-koi-id}" --branch main --commit-dirty=true
