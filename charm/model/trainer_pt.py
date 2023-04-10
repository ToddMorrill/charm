import copy
import time
import logging
import random

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import wandb

from .utils import get_optimizer


class Trainer():

    def __init__(self,
                 args,
                 model,
                 train_loader,
                 val_loader=None,
                 test_loader=None):
        self.args = args
        if 'wandb_project' in args and args.wandb_project is not None:
            self.run = wandb.init(project=args.wandb_project, config=self.args)
        self.model = model.to(args.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # record what splits are available
        self.splits = ['train']
        if self.val_loader is not None:
            self.splits.append('val')
        if self.test_loader is not None:
            self.splits.append('test')
        self.device = args.device
        self.epochs = args.epochs
        self._init_metrics()

    def _init_metrics(self):
        """Creates data structures for storing metrics."""
        self.metrics = {}
        for split in self.splits:
            self.metrics[split] = {
                'epoch_data_times': [],
                'epoch_batch_times': [],
                'epoch_loss_vals': [],
                'epoch_accuracies': [],
                'epoch_total_times': []
            }
        self._batch_metrics = {}
        for split in self.splits:
            self._reset_metrics(split)

    def _reset_metrics(self, split):
        """Resets all cumulative batch level metrics."""
        self._batch_metrics[split] = {
                'num_correct': 0,
                'num_examples': 0,
                'total_loss': 0,
                'data_times': [],
                'batch_times': [],
            }
            

    def log(self, msg):
        """Logs a message to the console."""
        # TODO: only do this on on RANK 0 node
        logging.info(msg)

    def _update_metrics(self, split, num_correct, num_examples, total_loss, data_time=None, batch_time=None):
        """Updates the batch level metrics."""
        batch_metrics = self._batch_metrics[split]
        batch_metrics['num_correct'] += num_correct
        batch_metrics['num_examples'] += num_examples
        batch_metrics['total_loss'] += total_loss
        # batch_metrics['data_times'].append(data_time)
        # batch_metrics['batch_times'].append(batch_time)

    def train_step(self, batch):
        """Runs a single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        # move data to device
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # TODO: implement multi-task loss
        outputs = self.model(**batch)
        loss = outputs.loss
        loss.backward()
        self.optimizer.step()

        preds = outputs.logits.argmax(dim=1)
        labels = batch['labels']
        # compute accuracy
        # TODO: determine if this is forcing a sync and slowing down the pipeline
        num_correct = (preds == labels).sum().detach().item()
        num_examples = len(preds)
        batch_accuracy = (num_correct / len(preds))
        loss = loss.detach().item()
        total_loss = loss * len(preds)
        self._update_metrics('train', num_correct, num_examples, total_loss)
        self._report(split='train')
        return loss, batch_accuracy

    def val_step(self, batch):
        """Runs a single evaluation step."""
        self.model.eval()
        # move data to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = self.model(**batch)
            loss = outputs.loss
            preds = outputs.logits.argmax(dim=1)
            labels = batch['labels']
            num_correct = (preds == labels).sum().detach().item()
            num_examples = len(preds)
            batch_accuracy = (num_correct / len(preds))
            loss = loss.detach().item()
            total_loss = loss * len(preds)
        
        self._update_metrics('val', num_correct, num_examples, total_loss)
        return loss, batch_accuracy

    def _report(self, split='train'):
        if self.args.reporting_steps == -1:
            return None
        # report metrics
        if self.global_step % self.args.reporting_steps == 0:
            # note that this reports the average batch metrics over all examples seen in the current epoch
            loss = self._batch_metrics[split]['total_loss'] / self._batch_metrics[split]['num_examples']
            accuracy = self._batch_metrics[split]['num_correct'] / self._batch_metrics[split]['num_examples']
            logging.info(
                f'{split} average batch metrics at (epoch:batch) ({self.epoch}:{self.global_step}) - Loss {loss:.2f} - Accuracy - {accuracy:.2f}'
            )

    def _record_metrics(self, split='train'):
        # self.metrics[split]['epoch_data_times'].append(self._epoch_data_times)
        # self.metrics[split]['epoch_batch_times'].append(
        #     self._epoch_batch_times)
        # going to ignore batch level loss information in favor over epoch losses
        # self.metrics[split]['epoch_loss_vals'].append(self._epoch_loss_vals)
        epoch_loss = self._batch_metrics[split]['total_loss'] / self._batch_metrics[split]['num_examples']
        epoch_accuracy = self._batch_metrics[split]['num_correct'] / self._batch_metrics[split]['num_examples']
        self.metrics[split]['epoch_loss_vals'].append(epoch_loss)
        self.metrics[split]['epoch_accuracies'].append(epoch_accuracy)

    def create_optimizer(self, ):
        """Creates an optimizer (and potentially a learning rate scheduler) for the model."""
        self.optimizer, self.lr_scheduler = get_optimizer(
            self.args, self.model, **self.args.optimizer_kwargs)

    def train(self):
        """Runs training and evaluation for the specified number of epochs."""
        self.global_step = 0
        self.epoch = 0
        for epoch in range(self.epochs):
            self.epoch = epoch
            epoch_start = time.perf_counter()
            # reset epoch level metrics
            self._reset_metrics('train')
            # train on training set
            for idx, batch in enumerate(self.train_loader):
                self.train_step(batch)

                # potentially evaluate on validation set
                if self.args.val_steps > 0 and self.global_step % self.args.val_steps == 0:
                    self._reset_metrics(split='val')
                    for val_idx, val_batch in enumerate(self.val_loader):
                        self.val_step(val_batch)
                    self._report(split='val')
                    self._record_metrics(split='val')
                
                self.global_step += 1

            # record metrics
            self._record_metrics(split='train')

    def get_metrics(self, split='train', return_df=True):
        metrics = copy.deepcopy(self.metrics[split])
        for key in ['epoch_data_times', 'epoch_batch_times']:
            raw_values = self.metrics[split][key]
            # sum over batch times
            metrics[key] = [sum(x) for x in raw_values]

        if return_df:
            df = pd.DataFrame(metrics)
            column_map = {
                'epoch_data_times': 'Epoch Data Loading Time',
                'epoch_batch_times': 'Epoch Training Time',
                'epoch_total_times': 'Epoch Total Time',
                'epoch_loss_vals': 'Epoch Average Loss',
                'epoch_accuracies': 'Epoch Average Accuracy'
            }
            df.rename(columns=column_map, inplace=True)
            df.index.name = 'Epoch'
            df.reset_index(inplace=True)
            return df

        return metrics