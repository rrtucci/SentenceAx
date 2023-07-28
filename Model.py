from my_globals import *
import pytorch_lightning as pl

class Model(pl.LightningModule):
    def __init__(self):
        super().__init__(**hparams)


    def forward(self, x):
        return output

    def training_step(self, batch, batch_index):

    def validation_step(self, batch, batch_idx):

    def validation_epoch_end(self, validation_step_outputs):

    def configure_optimizers(self):


