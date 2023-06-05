from typing import Any
from numpy.typing import NDArray

from pytorch_lightning.utilities.types import STEP_OUTPUT

import pytorch_lightning as pl
import numpy as np
from PIL import Image
from tempfile import NamedTemporaryFile


def make_image_from_rgb(
    rgb_pred:NDArray[np.float32], 
    rgb_gt:NDArray[np.float32]=None
        ) -> Image:

    image = (rgb_pred * 255.0).astype(np.uint8)
    if rgb_gt is not None:
        rgb_gt = (rgb_gt * 255.0).astype(np.uint8)
        image = np.concatenate((image, rgb_gt), axis=1)

    return Image.fromarray(image)


class LogRenders(pl.Callback):
    def __init__(
            self, img_h: int, chunk_size: int, img_w: int=None, 
            stages:list|set=None, log_every_n_images:int=None,
            log_every_n_epochs:int=None):
        """
        Logs rendered RGB images after a batch for all specified stages. This
        function assumes that chunks are square in shape.

        This callback is complicated because of chunking. We need to know when 
        enough chunks have accumulated to write an image, then we have to call 
        the appropriate functions to handle that.

        `img_h`: Height of the images 
        `chunk_size`: height of the chunk
        `img_w`: unused (may implement if rectangular images are needed)
        `stages`: a list of stages for which to log renders (e.g., ['train', 'val', 'test'])
        `log_every_n_images`: number of images to log. If you know the size of 
            your dataset, you can control how many images get logged. Recommended
            50 or 100 for nerf_synthetic datasets.
        `log_every_n_epochs`: specifies a logging frequency at the epoch level.
            Use this with log_every_n_images to get fine-grained control over 
            image-artifact logging.
        """
        self.stages = set(stages)

        self.gts = []
        self.preds = []

        self.img_h = img_h
        self.chunk_size = chunk_size

        self.img_w = img_w if img_w else self.img_h

        self.H_chunks = int(np.ceil(self.img_h / self.chunk_size))
        self.W_chunks = int(np.ceil(self.img_w / self.chunk_size))

        self.img_chunks_total = self.H_chunks * self.W_chunks

        self.fn_cache = {}
        
        self.log_every_n_epochs = log_every_n_epochs
        self.log_every_n_images = log_every_n_images if log_every_n_images else 1
        self.log_counter = 0 

    def collect_outputs(self, outputs):
        rgb_pred = outputs['rgb_pred'].detach().cpu()
        rgb_gt = outputs['rgb_gt'].detach().cpu()
        self.gts.append(rgb_gt)
        self.preds.append(rgb_pred)

    def build_rgbs(self, dataset):
        rgb_pred = dataset.image_from_chunks(self.preds)
        rgb_gt = dataset.image_from_chunks(self.gts)
        image = make_image_from_rgb(rgb_pred, rgb_gt)

        self.gts = []; self.preds = []
        return image
    
    def dump_image(self, outputs, phase, trainer):
        if phase in self.stages:
            self.collect_outputs(outputs)

            if len(self.gts) == len(self.preds) == self.img_chunks_total:
                self.log_counter += 1
                if self.log_counter % self.log_every_n_images == 0:
                    if phase not in self.fn_cache:
                        self.fn_cache[phase] = getattr(trainer.datamodule, f"{phase}_dataloader")().dataset.image_from_chunks
                        
                    rgb_pred = self.fn_cache[phase](self.preds)
                    rgb_gt = self.fn_cache[phase](self.gts)

                    image = make_image_from_rgb(rgb_pred.numpy(), rgb_gt.numpy())

                    with NamedTemporaryFile(suffix='.png', prefix=f"{phase}_render_") as tmp_file:
                        image.save(tmp_file)
                        trainer.logger.log_image(key=f"{phase}/render-test", images=[tmp_file.name])

                # Clear image
                self.gts = []; self.preds = []
    
    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.log_counter = 0
        self.gts = []; self.preds = []

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.log_counter = 0
        self.gts = []; self.preds = []

    def on_test_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.log_counter = 0 
        self.gts = []; self.preds = []
    
    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: STEP_OUTPUT | None, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        self.dump_image(outputs, "train", trainer)
    
    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: STEP_OUTPUT | None, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        self.dump_image(outputs, "val", trainer)

    def on_test_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: STEP_OUTPUT | None, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        self.dump_image(outputs, "test", trainer)

                
