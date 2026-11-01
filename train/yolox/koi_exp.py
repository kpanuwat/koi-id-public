"""YOLOX-tiny (Apache-2.0) experiment for koi: one class, boxes only. Used on Colab by pending/yolox/colab.sh.
TEST=<px> sets the export/eval input size (default 416)."""
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super().__init__()
        self.depth, self.width = 0.33, 0.375  # yolox_tiny
        self.input_size = (416, 416)
        t = int(os.environ.get("TEST", 416))
        self.test_size = (t, t)
        self.mosaic_scale = (0.5, 1.5)
        self.random_size = (10, 20)
        self.enable_mixup = False
        self.num_classes = 1
        self.data_dir = os.environ.get("DST", "/content/koiyx")
        self.train_ann, self.val_ann = "train.json", "val.json"
        self.max_epoch = int(os.environ.get("EPOCHS", 80))
        self.no_aug_epochs = 15
        self.warmup_epochs = 3
        self.eval_interval = 10
        self.data_num_workers = 4
        self.exp_name = "koi_yx"
