from typing import Any, Optional
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from tempfile import NamedTemporaryFile
from PIL import Image


def make_image_from_rgb(
    rgb_pred:NDArray[np.float32], 
    rgb_gt:NDArray[np.float32]=None
        ) -> Image:

    image = (rgb_pred * 255.0).astype(np.uint8)
    if rgb_gt is not None:
        rgb_gt = (rgb_gt * 255.0).astype(np.uint8)
        image = np.concatenate((image, rgb_gt), axis=1)

    return Image.fromarray(image)


class LogRenderedRGB(pl.Callback):
    def __init__(self, logging_frequency:int=1, stages:list|set=None):
        """
        Logs rendered RGB images after a batch for all specified stages.

        `logging_frequency`: 
        """
        self.logging_frequency = logging_frequency
        self.stages = set(stages)


    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: STEP_OUTPUT | None, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if 'train' in self.stages:  # and (batch_idx % self.logging_frequency == 0):
            rgb_pred = outputs['rgb_pred'].detach().cpu().numpy()
            rgb_gt = outputs['rgb_gt'].detach().cpu().numpy()

            image = make_image_from_rgb(rgb_pred, rgb_gt)

            with NamedTemporaryFile(suffix='.png', prefix="train_render_") as tmp_file:
                if (batch_idx % self.logging_frequency == 0):
                    image.save(tmp_file)
                    trainer.logger.log_image(key="train/render-test", images=[tmp_file.name])

            # image.save("/home/djones/test.png")

            plt.close()


    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: STEP_OUTPUT | None, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if 'val' in self.stages and (batch_idx & self.logging_frequency == 0):
            rgb_pred = outputs['rgb_pred'].detach().cpu().numpy()
            rgb_gt = outputs['rgb_gt'].detach().cpu().numpy()
            
            image = make_image_from_rgb(rgb_pred, rgb_gt)

            with NamedTemporaryFile(suffix='.png', prefix="val_render_") as tmp_file:
                image.save(tmp_file)
                trainer.logger.log_image(key="val/render-test", images=[tmp_file.name])

            plt.close()

    def on_test_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: STEP_OUTPUT | None, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if 'test' in self.stages and (batch_idx & self.logging_frequency == 0):
            rgb_pred = outputs['rgb_pred'].detach().cpu().numpy()
            rgb_gt = outputs['rgb_gt'].detach().cpu().numpy()

            image = make_image_from_rgb(rgb_pred, rgb_gt)

            with NamedTemporaryFile(suffix='.png', prefix="test_render_") as tmp_file:
                image.save(tmp_file)
                trainer.logger.log_image(key="test/render-test", images=[tmp_file.name])

            plt.close()



