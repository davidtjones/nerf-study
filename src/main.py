
from pytorch_lightning import Trainer
from nerf.OriginalNeRF import OriginalNeRF
from torchvision.transforms import Resize

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from util.callbacks import LogRenders
from util.data import NeRFDataModule


data_path = "../data/nerf_synthetic/lego"
devices = 1
batch_size = 1
ray_chunk_size = 50
img_height = 100
resampled_height = None

dm = NeRFDataModule(
    data_path,
    batch_size,
    ray_chunk_size,
    Resize(img_height, antialias=False)
)

model = OriginalNeRF()

logger = WandbLogger(project="nerf-study", entity="djones", save_dir="logging")

callbacks = [
    LogRenders(img_height, ray_chunk_size, stages={"train", "val"}, log_every_n_images=100),
    ModelCheckpoint(dirpath="logging", monitor='val/mse-loss')
]

trainer = Trainer(
    accelerator='gpu',
    logger=logger,
    max_epochs=65,
    callbacks=callbacks,
    default_root_dir="logging",
    devices=devices,
    strategy='ddp_find_unused_parameters_true' if devices > 1 else "auto"
)

trainer.fit(model, dm)
trainer.test(model, dm, chkpt_path='best')

