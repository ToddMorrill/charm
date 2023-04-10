"""Rewrite of the xlm-roberta-base model to use pytorch-lightning.

Examples:
    $ python -m charm.model.train_pl \
        --fast-dev-run \
        --model-dir /tmp/model

    # run in background
    $ nohup python -m charm.model.train_pl \
        > train.log 2>&1 &
    
"""
import argparse
import os
import math
from types import SimpleNamespace
import warnings

warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings(
    "ignore",
    ".*The `srun` command is available on your system but is not used.*")
warnings.filterwarnings(
    "ignore",
    ".*Starting from v1.9.0, `tensorboardX` has been removed as a dependency of the `pytorch_lightning` package*"
)

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
import transformers
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    DataCollatorWithPadding,
)

from charm.model.utils import get_circumpex_dataset, get_imdb_dataset, get_dataloaders

transformers.logging.set_verbosity_warning()
pd.options.mode.chained_assignment = None


class CircumplexClassifier(pl.LightningModule):

    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config.__dict__, ignore=['tokenizer'])
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass",
                                                    num_classes=len(
                                                        config.id2label))
        self.valid_accuracy = torchmetrics.Accuracy(task="multiclass",
                                                    num_classes=len(
                                                        config.id2label))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name_or_path,
            num_labels=len(config.id2label),
            id2label=config.id2label,
            label2id=config.label2id)
        self.tokenizer = tokenizer

    def forward(self,
                input_ids,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                labels=None):
        return self.model(input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          position_ids=position_ids,
                          head_mask=head_mask,
                          labels=labels)

    def training_step(self, batch, batch_idx):
        # input_ids = batch['input_ids']
        # attention_mask = batch['attention_mask']
        # labels = batch['labels']
        outputs = self(**batch)
        loss = outputs[0]
        self.log('train_loss',
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True,
                 sync_dist=True)

        # compute accuracy
        preds = outputs[1]
        labels = batch['labels']
        self.train_accuracy(preds, labels)
        self.log('train_acc',
                 self.train_accuracy,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=False,
                 logger=True,
                 sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        self.log('val_loss',
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)

        preds = outputs[1]
        labels = batch['labels']
        self.valid_accuracy(preds, labels)
        self.log('val_acc',
                 self.valid_accuracy,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        return loss

    def train_dataloader(self):
        return super().train_dataloader()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.config.learning_rate)
        # get a linear learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.num_warmup_steps,
            num_training_steps=self.config.num_train_steps)
        return [optimizer], [scheduler]


def main(args):
    # set up wandb
    os.environ['WANDB_PROJECT'] = 'social-orientation'

    np.random.seed(42)
    torch.manual_seed(42)
    config = {
        'model_name_or_path': 'xlm-roberta-base',
        'per_device_train_batch_size': 16,
        'val_check_interval': 100,
        'epochs': 20,
        'default_learning_rate': 5e-5,  # default that HF uses
        'learning_rate': 5e-5,
        'num_gpus': torch.cuda.device_count() if not args.fast_dev_run else 1,
        'debug_pct': 1,
        'wandb': True,
        'num_sanity_val_steps': 0,
        'overfit_batches': 0,
        'strategy': 'ddp',
        'ckpt_path': None,
    }
    config = SimpleNamespace(**config)

    if args.fast_dev_run:
        config.val_check_interval = None
        config.epochs = 2
        config.num_gpus = 1
        config.wandb = False
        config.debug_pct = 0.05
        config.num_sanity_val_steps = 2
        config.overfit_batches = 10
        config.strategy = None

    # https://arxiv.org/pdf/1706.02677.pdf
    # scale learning rate by data parallelism increase
    config.effective_batch_size = 3 * config.per_device_train_batch_size
    config.lr_multiplier = config.effective_batch_size / 32  # assuming 32 is the default batch size
    config.learning_rate = config.default_learning_rate * config.lr_multiplier  # Trainer defaults to 5e-5, with a linear decay and 0 warmup

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path,
                                              use_fast=False)

    # get the dataset
    if args.fast_dev_run and args.dataset != 'circumplex':
        # useful for sanity checking the system
        data = get_imdb_dataset(tokenizer, debug_pct=0.1)
    else:
        # get the Circumplex dataset
        data = get_circumpex_dataset(tokenizer, args.data_dir,
                                     config.debug_pct)

    train_dataset, val_dataset, id2label, label2id = data
    config.id2label = id2label
    config.label2id = label2id
    collate = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataloader, val_dataloader = get_dataloaders(config, collate,
                                                       train_dataset,
                                                       val_dataset)

    # get appoximate number of training steps
    batches_per_epoch = math.ceil(
        len(train_dataset) /
        (config.per_device_train_batch_size * config.num_gpus))
    config.num_train_steps = batches_per_epoch * config.epochs
    config.num_warmup_steps = config.num_train_steps // 10  # 10% warmup

    # set up the model
    model = CircumplexClassifier(config, tokenizer)
    # set up the trainer
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=args.model_dir,
        # filename='best-checkpoint',
        save_last=True,
        save_top_k=1,
        mode='min',
        every_n_train_steps=config.val_check_interval,
        verbose=True)
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10, # 10 validation steps without improvement
        mode='min',
        verbose=True,
        check_on_train_epoch_end=False,  # checks every time validation is run 
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [early_stop, lr_monitor, checkpoint_callback]
    if config.wandb:
        wandb_logger = WandbLogger(project='social-orientation')
    trainer = Trainer(
        accelerator='gpu',
        devices=config.num_gpus,
        max_epochs=config.epochs,
        num_sanity_val_steps=config.num_sanity_val_steps,
        val_check_interval=config.val_check_interval,
        overfit_batches=config.overfit_batches,
        callbacks=callbacks,
        logger=wandb_logger if config.wandb else None,
    )

    if args.resume:
        # resume_path = os.path.join(args.model_dir, 'best.ckpt')
        # if os.path.exists(resume_path):
        config.ckpt_path = 'last'
        print(f'Resuming training from {config.ckpt_path}')

    trainer.fit(model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
                ckpt_path=config.ckpt_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast-dev-run', action='store_true', default=False)
    parser.add_argument(
        '--data-dir',
        type=os.path.expanduser,
        default='~/Documents/data/charm/transformed/circumplex')
    parser.add_argument(
        '--model-dir',
        type=os.path.expanduser,
        default='~/Documents/data/charm/models/xlm-roberta-base-pl')
    parser.add_argument('--resume',
                        action='store_true',
                        default=False,
                        help='Resume training from a checkpoint.')
    parser.add_argument(
        '--dataset',
        type=str,
        default='circumplex',
        help='Which dataset to use. Options: [circumplex, imdb]')
    args = parser.parse_args()
    main(args)
