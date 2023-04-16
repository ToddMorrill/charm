"""Rewrite of the classifier training script in native PyTorch.

Examples:
    $ python -m charm.model.train_pt \
        --wandb-project change-point \
        --epochs 20 \
        --log-level INFO \
        --seed 10 \
        --dataset change-point \
        --data-dir /mnt/swordfish-pool2/ccu/transformed/change-point \
        --model-dir /mnt/swordfish-pool2/ccu/models/change-point-class-reweight \
        --window-size 10 \
        --class-weights 1 50
    
    $ python -m charm.model.train_pt \
        --batch-size 64 \
        --wandb-project social-orientation \
        --epochs 20 \
        --log-level INFO \
        --seed 10 \
        --val-steps 500 \
        --data-dir /mnt/swordfish-pool2/ccu/transformed/circumplex \
        --model-dir /mnt/swordfish-pool2/ccu/models/xlm-roberta-base-pt

    # distributed training
    $ nohup torchrun --nproc_per_node=3 --nnodes=1 \
        -m charm.model.train_pt \
        --distributed --batch-size 64 --log-level INFO \
        --val-steps 200 \
        --epochs 20 --wandb-project social-orientation \
        --seed 10 \
        --data-dir /mnt/swordfish-pool2/ccu/transformed/circumplex \
        --model-dir /mnt/swordfish-pool2/ccu/models/xlm-roberta-base-pt \
        > train.log 2>&1 &
    
    $ CUDA_VISIBLE_DEVICES=0 nohup python -m charm.model.train_pt \
        --wandb-project change-point --epochs 20 \
        --log-level INFO --seed 10 --dataset change-point \
        --data-dir /mnt/swordfish-pool2/ccu/transformed/change-point \
        --model-dir /mnt/swordfish-pool2/ccu/models/change-point-impact-scalar \
        --class-weight 25  --window-size 10 --impact-scalar > train_impact_scalar.log 2>&1 &
    
    $ CUDA_VISIBLE_DEVICES=0 nohup python -m charm.model.train_pt \
        --wandb-project change-point --epochs 20 \
        --log-level INFO --seed 10 --dataset change-point \
        --data-dir /mnt/swordfish-pool2/ccu/transformed/change-point \
        --model-dir /mnt/swordfish-pool2/ccu/models/change-point-triplet-loss \
        --class-weight 25  --window-size 10 --triplet-loss > train_triplet_loss.log 2>&1 &
    
    $ CUDA_VISIBLE_DEVICES=1 nohup python -m charm.model.train_pt \
        --wandb-project change-point --epochs 20 \
        --log-level INFO --seed 10 --dataset change-point \
        --data-dir /mnt/swordfish-pool2/ccu/transformed/change-point \
        --model-dir /mnt/swordfish-pool2/ccu/models/change-point-high-reweight \
        --class-weight 50  --window-size 10 > train_high_reweight.log 2>&1 &
    
    $ CUDA_VISIBLE_DEVICES=3 nohup python -m charm.model.train_pt \
        --wandb-project change-point --epochs 20 \
        --log-level INFO --seed 10 --dataset change-point \
        --data-dir /mnt/swordfish-pool2/ccu/transformed/change-point \
        --model-dir /mnt/swordfish-pool2/ccu/models/change-point \
        --class-weight 25  --window-size 10 > train.log 2>&1 &
    
    $ CUDA_VISIBLE_DEVICES=4 nohup python -m charm.model.train_pt \
        --wandb-project change-point --epochs 20 \
        --log-level INFO --seed 10 --dataset change-point \
        --data-dir /mnt/swordfish-pool2/ccu/transformed/change-point \
        --model-dir /mnt/swordfish-pool2/ccu/models/change-point-social-orientation \
        --class-weight 25  --window-size 10 --social-orientation > train_social_orientation.log 2>&1 &
    
    $ CUDA_VISIBLE_DEVICES=5 nohup python -m charm.model.train_pt \
        --wandb-project change-point --epochs 20 \
        --log-level INFO --seed 10 --dataset change-point \
        --data-dir /mnt/swordfish-pool2/ccu/transformed/change-point \
        --model-dir /mnt/swordfish-pool2/ccu/models/change-point-medium-reweight \
        --class-weight 12  --window-size 10 > train_medium_reweight.log 2>&1 &
    
    $ CUDA_VISIBLE_DEVICES=6 nohup python -m charm.model.train_pt \
        --wandb-project change-point --epochs 20 \
        --log-level INFO --seed 10 --dataset change-point \
        --data-dir /mnt/swordfish-pool2/ccu/transformed/change-point \
        --model-dir /mnt/swordfish-pool2/ccu/models/change-point-light-reweight \
        --class-weight 2  --window-size 10 > train_light_reweight.log 2>&1 &
    
    $ CUDA_VISIBLE_DEVICES=7 nohup python -m charm.model.train_pt \
        --wandb-project change-point --epochs 20 \
        --log-level INFO --seed 10 --dataset change-point \
        --data-dir /mnt/swordfish-pool2/ccu/transformed/change-point \
        --model-dir /mnt/swordfish-pool2/ccu/models/change-point-light-reweight-high-window-impact-scalar \
        --class-weight 2  --window-size 20 --impact-scalar > train_light_reweight_high_window_impact_scalar.log 2>&1 &
TODOs:
- figure out how to set the wandb loss as the best recorded loss instead of the ending loss
- determine if it makes sense to resume training from a checkpoint with a lower learning rate
- explore different learning rate schedulers that respond to the validation loss (i.e. lower on plateau)
"""
import sys
import logging
import signal

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import transformers
from transformers import AutoModelForSequenceClassification
import pandas as pd

from .args import parse_args
from .trainer_pt import Trainer
from .utils import get_data, dist_log
from .model import XLMRClassificationPlusTripletLoss

transformers.logging.set_verbosity_warning()
pd.options.mode.chained_assignment = None


def handler(signum, frame):
    """Shuts down the distributed training when ctrl+c signal is detected."""
    logging.info('Ctrl+C detected, Shutting down distributed training.')
    dist.destroy_process_group()
    sys.exit(0)


signal.signal(signal.SIGINT, handler)

def hyperparameter_search(args):
    """Run hyperparameter search."""
    # TODO: implement this in a way that is compatible with the existing code
    pass

def pipeline(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.local_rank = None
    args.world_size = None
    # set up distributed training
    if args.distributed:
        dist.init_process_group("gloo")
        rank = dist.get_rank()
        args.local_rank = rank
        world_size = dist.get_world_size()
        args.world_size = world_size
        device_id = rank % torch.cuda.device_count()
        args.device = torch.device(f'cuda:{device_id}')
        dist_log(args, 'Distributed training enabled.')
        logging.info(f'Rank: {rank}, world size: {world_size}')

    dist_log(args, f'Running pipeline with the following arguments:\n{args}')
    dist_log(args,
             f'Training for {args.epochs} epochs on device: {args.device}.')

    # load data
    train_loader, val_loader, test_loader, args = get_data(args)

    # get appoximate number of training steps
    batches_per_epoch = len(train_loader)
    args.num_train_steps = batches_per_epoch * args.epochs
    args.num_warmup_steps = args.num_train_steps // 20  # 20% warmup

    # TODO: adjust learning rate as a function of number of devices and batch size

    if args.triplet_loss or args.impact_scalar or args.social_orientation or (args.class_weight is not None):
        model = XLMRClassificationPlusTripletLoss.from_pretrained(
            'xlm-roberta-base',
            num_labels=len(args.id2label),
            id2label=args.id2label,
            label2id=args.label2id)
        # this will enable things like triplet loss, impact scalar, social orientation, and class weighting
        model.add_args(args)
        if args.class_weight is not None:
            dist_log(args, f'Using class weights: {args.class_weight}')
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            num_labels=len(args.id2label),
            id2label=args.id2label,
            label2id=args.label2id)
    # model.to(args.device)
    logging.info(f'Model loaded on device: {args.device}.')
    # train model
    trainer = Trainer(args, model, train_loader, val_loader)
    trainer.train()

    if args.distributed:
        logging.debug(
            f'Shutting down distributed training on rank: {args.local_rank}.')
        dist.barrier()
        logging.debug(
            f'Shutting down distributed training on rank: {args.device}.')
        dist.destroy_process_group()


def main(args):
    pipeline(args)


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    # set args.log_level
    args_log_level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    logging.basicConfig(datefmt='%Y-%m-%d %I:%M:%S',
                        format='%(asctime)s %(levelname)s %(message)s',
                        level=args_log_level_map[args.log_level])

    main(args)