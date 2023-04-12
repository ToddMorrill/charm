import argparse
import logging
import os

import torch


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-dir',
        type=os.path.expanduser,
        default='~/Documents/data/charm/transformed/circumplex')
    parser.add_argument(
        '--model-dir',
        type=os.path.expanduser,
        default='~/Documents/data/charm/models/xlm-roberta-base-pt')
    parser.add_argument('--resume',
                        action='store_true',
                        default=False,
                        help='Resume training from a checkpoint.')
    parser.add_argument('--fast-dev-run', action='store_true', default=False, help='Run 1 batch of train, val, test for debugging purposes.')
    parser.add_argument(
        '--dataset',
        type=str,
        default='circumplex',
        help='Which dataset to use. Options: [circumplex, imdb]')
    parser.add_argument('--model-name-or-path',
                        type=str,
                        default='xlm-roberta-base',
                        help='The name or path of the model to use.')
    parser.add_argument(
        '--disable-cuda',
        action='store_true',
        help='If passed, the code will use the CPU instead of the GPU.')
    parser.add_argument(
        '--num-dataloader-workers',
        default=2,
        help='The number of workers for the train and test dataloaders.')
    parser.add_argument(
        '--optimizer',
        default='AdamW',
        choices=[
            'SGD', 'SGD-Nesterov', 'Adagrad', 'Adadelta', 'Adam', 'AdamW'
        ],
        help=
        'The optimizer used to train the network. Must be one of {SGD, SGD-Nesterov, Adagrad, Adadelta, Adam, AdamW}.'
    )
    parser.add_argument(
        '--lr-scheduler',
        default='linear-with-warmup',
        choices=['linear-with-warmup'],
        help=
        'The learning rate scheduler to employ. Must be one of {linear-with-warmup}.'
    )
    parser.add_argument(
        '--disable-batch-norm',
        action='store_true',
        help='If passed, the model will not use batch norm layers.')
    parser.add_argument('--epochs',
                        type=int,
                        default=10,
                        help='The number of training epochs.')
    parser.add_argument(
        '--debug-pct',
        type=float,
        default=1.0,
        help=
        'Subsets the dataset by the specified percent (e.g. 0.1) for rapid development. Defaults to 1.0.'
    )
    parser.add_argument('--batch-size',
                        type=int,
                        default=16,
                        help='The batch size to use for training.')
    parser.add_argument('--wandb-project',
                        type=str,
                        help='The name of the wandb project to log to.')
    parser.add_argument('--lr',
                                            type=float,
                                            default=5e-5,
                                            help='The learning rate.')
    parser.add_argument('--reporting-steps', type=int, default=100, help='The number of steps between each reporting step.')
    parser.add_argument('--val-steps', type=int, default=100, help='The number of steps between evaluation.')
    parser.add_argument('--distributed', action='store_true', default=False, help='If passed, the code will run in distributed mode using DDP.')
    parser.add_argument('--log-level', type=str, default='INFO', help='The logging level to use. Must be one of {DEBUG, INFO, WARNING, ERROR, CRITICAL}.')
    parser.add_argument('--num-checkpoints', type=int, default=2, help='The number of checkpoints to keep. This will always attempt to keep the best checkpoint first.')
    parser.add_argument('--early-stopping-patience', type=int, default=10, help='The number of evaluations to wait before early stopping.')
    parser.add_argument('--seed', type=int, default=42, help='The random seed to use.')
    parser.add_argument('--triplet-loss', action='store_true', default=False, help='If passed, the model will use a triplet loss.')
    args = parser.parse_args(args)
    # TODO: do we want to adjust this by the effective batch size?
    args.optimizer_kwargs = {'lr': args.lr}
    args.device = None
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    return args