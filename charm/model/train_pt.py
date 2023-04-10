"""Rewrite of the classifier training script in native PyTorch.

Examples:
    $ python train_pt.py
"""
import sys
import logging

import torch
from torch import nn
import torch.optim as optim
from transformers import AutoModelForSequenceClassification
import pandas as pd

from .args import parse_args
from .trainer_pt import Trainer
from .utils import get_data

logging.basicConfig(datefmt='%Y-%m-%d %I:%M:%S',
                    format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG)


def pipeline(args):
    logging.info(
        f'Training for {args.epochs} epochs on device: {args.device}.')
    # load data

    train_loader, val_loader, args = get_data(args)
    # get appoximate number of training steps
    batches_per_epoch = len(train_loader)
    args.num_train_steps = batches_per_epoch * args.epochs
    args.num_warmup_steps = args.num_train_steps // 10  # 10% warmup

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path, num_labels=len(args.id2label),
            id2label=args.id2label,
            label2id=args.label2id)
    # train model
    trainer = Trainer(args, model, train_loader, val_loader)
    # configure the optimizer
    trainer.create_optimizer()
    
    trainer.train()


def main(args):
    metrics = pipeline(args)
    logging.info(f'Pipeline metrics:\n{metrics}')


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)