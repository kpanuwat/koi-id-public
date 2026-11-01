#!/bin/bash
# YOLOX-tiny koi detector on Colab (L4), same reviewed data as the hard-negative RF-DETR run. Run from the repo root.
#   pending/yolox/colab.sh run     # new session, upload, build data/koi2 (SAM2) -> koiyx, train, export ONNX 416 + 320
#   pending/yolox/colab.sh down    # download ONNX + best ckpt + log to train/runs/yx, stop the session
set -euo pipefail
S=koi-yx
EPOCHS=${EPOCHS:-80}
TMP=${CLAUDE_JOB_DIR:-/tmp}/tmp
case "${1:-}" in
run)
  tar -czf "$TMP/hn.tgz" -C pending/hardneg frames final.json --transform 's,^,hn/,'
  tar -czf "$TMP/koi.tgz" -C train data/koi/train data/koi/valid
  colab new -s $S --gpu L4
  printf "import os\nos.makedirs('/content/parts', exist_ok=True)\n" > "$TMP/mk.py"; colab exec -s $S -f "$TMP/mk.py"
  # uploads of ~100 MB fail with HTTP 500: send 40 MB parts, joined on the runtime
  for f in hn.tgz koi.tgz; do
    rm -f "$TMP"/part_*; split -b 40M "$TMP/$f" "$TMP/part_"
    for p in "$TMP"/part_*; do colab upload -s $S "$p" /content/parts/$f.$(basename $p); done
  done
  for f in train/colab_hardneg.py pending/yolox/koi_exp.py pending/yolox/to_yolox.py; do colab upload -s $S $f /content/$(basename $f); done
  cat > "$TMP/yx_setup.py" <<'EOF'
import subprocess
subprocess.run('''cd /content && for f in hn.tgz koi.tgz; do cat parts/$f.part_* > $f; done && rm -r parts && tar xzf hn.tgz && tar xzf koi.tgz
pip -q install transformers==5.20.2 loguru thop tabulate pycocotools ninja onnx
git clone -q https://github.com/Megvii-BaseDetection/YOLOX && cd YOLOX && pip -q install --no-deps --no-build-isolation -e .
# torch >= 2.9 has no torch.onnx._export: use the TorchScript exporter explicitly
sed -i 's/torch.onnx._export(/torch.onnx.export(/; s/        opset_version=args.opset,/        opset_version=args.opset, dynamo=False,/' tools/export_onnx.py
wget -q https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth''', shell=True, check=True, executable='/bin/bash')
print('setup done')
EOF
  colab exec -s $S --timeout 1200 -f "$TMP/yx_setup.py"
  # everything in the kernel FOREGROUND: a detached process leaves the kernel idle and Colab reclaims the runtime
  cat > "$TMP/yx_run.py" <<EOF
import subprocess
r = subprocess.run('''set -e; cd /content; mkdir -p out
python colab_hardneg.py build
python to_yolox.py
cd YOLOX; export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 EPOCHS=$EPOCHS
python tools/train.py -f /content/koi_exp.py -d 1 -b 32 --fp16 -c yolox_tiny.pth
for t in 416 320; do TEST=\$t python tools/export_onnx.py -f /content/koi_exp.py -c YOLOX_outputs/koi_yx/best_ckpt.pth --output-name /content/out/koi_yx\$t.onnx --opset 13 --no-onnxsim --decode_in_inference; done
cp YOLOX_outputs/koi_yx/best_ckpt.pth YOLOX_outputs/koi_yx/train_log.txt /content/out/''', shell=True, executable='/bin/bash',
    stdout=open('/content/yx.log', 'w'), stderr=subprocess.STDOUT)
print('EXIT', r.returncode)
EOF
  colab exec -s $S --timeout 14400 -f "$TMP/yx_run.py"
  ;;
down)
  mkdir -p train/runs/yx
  for f in koi_yx416.onnx koi_yx320.onnx best_ckpt.pth train_log.txt; do colab download -s $S /content/out/$f train/runs/yx/$f || true; done
  colab download -s $S /content/yx.log train/runs/yx/yx.log || true
  colab stop -s $S
  ;;
*) echo "usage: $0 run|down"; exit 1 ;;
esac
