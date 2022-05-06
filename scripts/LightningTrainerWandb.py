import sys
sys.path.insert(0, '../scripts/')

import wandb
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from helpers import miscellaneous as misc
from helpers import preprocessing2d as prep
from data_loader import MRIDataset
from loss_functions import get_optimizer, get_criterion
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import StochasticWeightAveraging
import torchmetrics
import monai
from datetime import datetime

#---------------------------------------

#maybe make config parameters available as script parameter (for example, -bs 256 to make batch size 256 or -dim 1 for dimension 1 in 2D slices
CONFIG = misc.get_config()

DEVICE = CONFIG['DEVICE']
LEARNING_RATE = CONFIG['LEARNING_RATE']
AUTO_LEARNING_RATE = CONFIG['AUTO_LEARNING_RATE'] #not used
BATCH_SIZE = CONFIG['BATCH_SIZE']
AUTO_BATCH_SIZE = CONFIG['AUTO_BATCH_SIZE'] #not used
EPOCHS = CONFIG['EPOCHS']
NUM_WORKERS = CONFIG['NUM_WORKERS']
TRANSFORMER = CONFIG['TRANSFORMER']
CRITERION = CONFIG['CRITERION']
OPTIMIZER = CONFIG['OPTIMIZER']
TRAIN_SET = '../' + CONFIG['TRAIN_LABELS_DIR']
TEST_SET = '../' + CONFIG['TEST_LABELS_DIR']
RAW_DATA = '../' + CONFIG['FLATTENED_DATA_DIR']
PLOT_DIR = '../' + CONFIG['PLOT_DIR_BINARY']
DIMENSION = CONFIG['DIMENSION']
NSLICE = CONFIG['NSLICE']
WANDB_USER = CONFIG['WANDB_USER']

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class_names = ['CN', 'MCI', 'AD']

class MRIDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_paths):
        super().__init__()
        self.batch_size = batch_size
        self.data_paths = data_paths
        self.train_transform, self.val_transform = prep.get_transformer(TRANSFORMER)
        self.train_set = None
        self.val_set = None

    def setup(self, stage=None):
        self.train_set = MRIDataset(self.data_paths['train_dir'], transform=self.train_transform, dimension=DIMENSION, nslice=NSLICE)
        self.val_set = MRIDataset(self.data_paths['val_dir'], transform=self.val_transform, dimension=DIMENSION, nslice=NSLICE)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=NUM_WORKERS, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=NUM_WORKERS, shuffle=False)

data = MRIDataModule(
    batch_size= BATCH_SIZE,
    data_paths = {'train_dir': TRAIN_SET,
                 'val_dir': TEST_SET},
)

class Model(pl.LightningModule):
    def __init__(self, net, criterion, learning_rate, optimizer_class):
        super().__init__()
        self.lr = learning_rate
        self.net = net
        self.criterion = criterion
        self.optimizer_class = optimizer_class
        
        #log hyperparameters
        self.save_hyperparameters(ignore=["net"])
        
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        return optimizer

    def prepare_batch(self, batch):
        return batch['images'], batch['labels']

    def infer_batch(self, batch):
        x, y = self.prepare_batch(batch)
        y_hat = self.net(x)
        return y_hat, y

    def training_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        
        self.train_acc(torch.argmax(y_hat, dim=1), torch.argmax(y, dim=1))
        self.log('train_acc', self.train_acc, prog_bar = True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss.item(), on_step=False, prog_bar=True, on_epoch=True)
        
        self.val_acc(torch.argmax(y_hat, dim=1), torch.argmax(y, dim=1))
        self.log('val_acc', self.val_acc, on_step=False, prog_bar = True, on_epoch=True)
        
        return loss
    
    def validation_epoch_end(self, val_step_outputs):
        dummy_input = torch.zeros((1, 1, 150,150), device = device)
        model_filename = "model_final.onnx"
        torch.onnx.export(self.net, dummy_input, model_filename)
        wandb.save(model_filename)

#add XAI elements here for XAI after each epoch
class ImagePredictionLogger(pl.Callback):
    def __init__(self, val_samples, num_samples=4):
        super().__init__()
        self.val_imgs, self.val_labels = val_samples
        self.val_imgs = self.val_imgs[:num_samples]
        self.val_labels = self.val_labels[:num_samples]
        
    def on_validation_epoch_end(self, trainer, pl_module):
        val_imgs = self.val_imgs.to(device = pl_module.device)
        
        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, -1)
        
        trainer.logger.experiment.log({
            "examples": [wandb.Image(x, caption=f"Pred: {pred}, Label:{y}") for x, pred, y in zip(val_imgs, preds, self.val_labels)],
            "global_step": trainer.global_step
        })

if __name__ == '__main__':
    
    data.setup()
    print("Training Set Size: ", len(data.train_set))
    print("Validation Set Size: ", len(data.val_set))
    samples = data.val_dataloader()
    
    early_stopping = pl.callbacks.early_stopping.EarlyStopping(
        monitor="train_loss"
    )

    wandb_logger = WandbLogger(project="mlmodels", entity="brain-health")

    wandb.init(project="mlmodels", entity="brain-health")

    trainer = pl.Trainer(
        max_epochs = EPOCHS,
        gpus=1,
        logger = wandb_logger,
        precision=16,
        log_every_n_steps=10,
        callbacks=[early_stopping], #StochasticWeightAveraging(swa_lrs=1e-2)], #ImagePredictionLogger(samples)],
        gradient_clip_val=0.5,
        auto_scale_batch_size = AUTO_BATCH_SIZE,
        auto_lr_find="lr"
    )
    trainer.logger._default_hp_metric = False

    conv = monai.networks.blocks.Convolution(
        dimensions=2,
        in_channels=1,
        out_channels=1,
        adn_ordering="ADN",
        act=("prelu", {"init": 0.2}),
        dropout=0.1
    )

    cnn_to_mlp = torch.nn.Sequential(
      torch.nn.Flatten(1, -1),
      torch.nn.Linear(150*150, 32),
      torch.nn.ReLU(),
      torch.nn.Linear(32, 3),  
      torch.nn.Softmax(dim=1)
    )

    NET = torch.nn.Sequential(
        conv,
        cnn_to_mlp
    )

    criterion = get_criterion()
    optimizer = get_optimizer(NET, CONFIG)

    model = Model(
        net=NET,
        criterion= F.cross_entropy,
        learning_rate=LEARNING_RATE,
        optimizer_class=torch.optim.AdamW,
    )
    trainer.tune(model, datamodule=data)
    
    if AUTO_LEARNING_RATE:
        #not working yet:
        """lr_finder = trainer.tuner.lr_find(model)

        # Results can be found in
        print(lr_finder.results)

        # Pick point based on plot, or get suggestion
        model.hparams.learning_rate = lr_finder.suggestion()"""
    
    start = datetime.now()
    print('[INFO] Training started at', start)
    trainer.fit(model=model, datamodule=data)
    print('[INFO] Training duration:', datetime.now() - start)

    print("[INFO] Finished Training")