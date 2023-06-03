
from pytorch_lightning import Trainer
from nerf.OriginalNeRF import OriginalNeRF
from torchvision.transforms import Resize

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from util.callbacks import LogRenderedRGB
from util.data import NeRFDataModule


data_path = "../data/nerf_synthetic/lego"

dm = NeRFDataModule(data_path, Resize(100, antialias=False))

model = OriginalNeRF()

logger = WandbLogger(project="nerf-study", entity="djones", save_dir="logging")

callbacks = [
    LogRenderedRGB(50, {"val"}),
    ModelCheckpoint(dirpath="logging", monitor='val/mse_loss')
]

trainer = Trainer(
    accelerator='gpu',
    logger=logger,
    max_epochs=1000,
    callbacks=callbacks,
    default_root_dir="logging"
    devices=1
)

trainer.fit(model, dm)
trainer.test(model, dm, chkpt_path='best')

