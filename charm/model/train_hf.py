"""This module is responsible for training the communication change model.

Examples:
    # The following approach uses all GPUs with the DataParallel wrapper.
    $ python -u -m charm.model.train_hf \
        --data-dir ~/Documents/data/charm/transformed/circumplex \
        --model-dir /tmp/imdb \
        --debug-pct 0.1 \
        --no-wandb \
        --imdb
    # The following uses DistributedDataParallel with 3 GPUs.
    $ accelerate launch \
        -m charm.model.train_hf \
        --data-dir ~/Documents/data/charm/transformed/circumplex \
        --model-dir ~/Documents/data/charm/models/xlm-roberta-base \
        --debug-pct 0.1 \
        --no-wandb
    # run in background
    $ nohup accelerate launch \
        -m charm.model.train_hf \
        --data-dir ~/Documents/data/charm/transformed/circumplex \
        --model-dir ~/Documents/data/charm/models/xlm-roberta-base \
        > train.log 2>&1 &
"""
import argparse
from copy import deepcopy
import os
import warnings
from types import SimpleNamespace

warnings.filterwarnings(
    'ignore',
    message=
    'Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.'
)
warnings.filterwarnings(
    'ignore',
    message=
    'This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning'
)

import numpy as np
import pandas as pd
import torch
import transformers
from evaluate import load
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, EarlyStoppingCallback,
                          Trainer, TrainingArguments)
import wandb

from .utils import CircumplexDataset

transformers.logging.set_verbosity_warning()
pd.options.mode.chained_assignment = None


def compute_metrics(eval_pred):
    acc = load('accuracy')
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return acc.compute(predictions=predictions, references=labels)


def early_stopping(dataset, per_device_train_batch_size, num_devices,
                   eval_steps):
    # set patience to be roughly 1 epoch
    iters_per_epoch = len(dataset) // (per_device_train_batch_size *
                                       num_devices)
    num_evals_per_epoch = iters_per_epoch // eval_steps
    patience = num_evals_per_epoch * 1
    early_stop = EarlyStoppingCallback(early_stopping_patience=patience)
    return early_stop



class EvalTrainCallback(transformers.TrainerCallback):

    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_evaluate(self, args, state, control, **kwargs):
        # breakpoint()
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.log(
                self._trainer.evaluate(
                    eval_dataset=self._trainer.train_dataset,
                    metric_key_prefix="train"))
            return control_copy


class CustomTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # log accuracy on train set
        if "labels" in inputs:
            preds = outputs.logits.detach()
            acc = ((preds.argmax(axis=1) == inputs["labels"]).type(
                torch.float).mean().item())
            self.log({"accuracy": acc})

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            # if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
            #     loss = self.label_smoother(outputs, labels, shift_labels=True)
            # else:
            loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


def get_sweep_config(args):
    sweep_config = {
        'method': 'random',
        "metric": {"name": "eval/loss", "goal": "minimize"},
    }
    # hyperparameters
    parameters_dict = {
        'epochs': {
            'value': 5
            },
        'per_device_train_batch_size': {
            'values': [16, 24]
            },
        'lr': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-3
        },
        'weight_decay': {
            'values': [0.0, 0.01, 0.1, 0.2]
        },
        'model': {
            'values': ['distilbert-base-multilingual-cased', 'xlm-roberta-base', 'bert-base-chinese']
        },
        'eval_log_steps': {'value': 100},
        'num_gpus': {'value': torch.cuda.device_count()},
        'debug_pct': {'value': args.debug_pct},
        'wandb': {'value': not args.no_wandb},

    }
    sweep_config['parameters'] = parameters_dict
    return sweep_config

def main(args):
    # set up wandb
    os.environ['WANDB_PROJECT'] = 'social-orientation'

    np.random.seed(42)
    torch.manual_seed(42)
    # model = 'xlm-roberta-base'
    # model = 'distilbert-base-uncased'
    # model = 'bert-base-chinese'
    hyperparams = {
        'model': 'xlm-roberta-base',
        'per_device_train_batch_size': 16,
        'eval_log_steps': 100,
        'epochs': 20,
        'lr': 5e-5,
        'num_gpus': torch.cuda.device_count(),
        'debug_pct': args.debug_pct,
        'wandb': not args.no_wandb,
    }
    tokenizer = AutoTokenizer.from_pretrained(hyperparams['model'],
                                              use_fast=False)
    # get the dataset
    if args.imdb:
        # useful for sanity checking the system
        data = get_imdb_dataset(tokenizer, args.debug_pct)
    else:
        # get the Circumplex dataset
        data = get_circumpex_dataset(tokenizer, args.data_dir, args.debug_pct)

    train_dataset, val_dataset, id2label, label2id = data
    collate = DataCollatorWithPadding(tokenizer=tokenizer)

    # this is really bad practice, probably need a class to persist the data
    # and regenerate the model under each new configuration
    def train(config=None):
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model,
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id)

        train_batch_size = config.per_device_train_batch_size
        eval_log_steps = config.eval_log_steps
        if config.debug_pct < 1:
            eval_log_steps = 10

        num_train_epochs = config.epochs
        num_train_steps = (
            len(train_dataset) //
            (train_batch_size * config.num_gpus)) * num_train_epochs
        num_warmup_steps = num_train_steps // 10  # 10% warmup
        # https://arxiv.org/pdf/1706.02677.pdf
        # scale learning rate by data parallelism increase
        effective_batch_size = 3 * train_batch_size
        lr_multiplier = effective_batch_size / 32  # assuming 32 is the default batch size
        lr = config.lr * lr_multiplier  # Trainer defaults to 5e-5, with a linear decay and 0 warmup
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        # get a linear learning rate scheduler
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps)
        report_to = 'wandb' if config.wandb else 'none'
        training_args = TrainingArguments(
            output_dir=args.model_dir,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=train_batch_size * 2,
            gradient_accumulation_steps=1,
            # learning_rate=lr,
            # TODO: understand this parameter
            overwrite_output_dir=True,
            evaluation_strategy='steps',
            eval_steps=eval_log_steps,
            logging_steps=eval_log_steps,
            dataloader_num_workers=config.num_gpus,
            # num_epochs determined time average time to converge in experiments
            num_train_epochs=num_train_epochs,
            save_total_limit=2,
            load_best_model_at_end=True,
            save_strategy='steps',
            save_steps=eval_log_steps,
            seed=42,
            report_to=report_to,
        )

        early_stop = early_stopping(train_dataset, train_batch_size,
                                    config.num_gpus, eval_log_steps)
        # early_stop = None
        callbacks = [early_stop] if early_stop else None
        trainer = CustomTrainer(
            model,
            training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collate,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=(optimizer, scheduler),
        )
        # trainer.add_callback(EvalTrainCallback(trainer))
        trainer_results = trainer.train()

    config = SimpleNamespace(**hyperparams)

    def train_wandb(config=None):
        with wandb.init(config=config, group='ddp'):
            # set sweep configuration
            config = wandb.config
            train(config)
    
    if args.sweep:
        sweep_config = get_sweep_config(args)
        sweep_id = wandb.sweep(sweep_config, project='social-orientation')
        wandb.agent(sweep_id, train_wandb, count=10)
        wandb.finish()
    else:
        train(config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',
                        type=str,
                        help='Directory where the raw data data is stored.')
    parser.add_argument('--model-dir',
                        type=str,
                        help='Directory where the model will be stored.')
    parser.add_argument(
        '--debug-pct',
        type=float,
        default=1,
        help=
        'Percentage of data to use for debugging. (default: 1, use all data).')
    parser.add_argument('--no-wandb',
                        action='store_true',
                        default=False,
                        help='Disable wandb logging. (default: False)')
    parser.add_argument(
        '--imdb',
        action='store_true',
        default=False,
        help='Use the IMDB dataset to sanity check the system. (default: False)'
    )
    parser.add_argument('--sweep', action='store_true', default=False, help='Run a hyperparameter sweep. (default: False)')
    args = parser.parse_args()
    main(args)