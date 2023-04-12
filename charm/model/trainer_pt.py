"""Trainer class for PyTorch models.

TODO:
- checkpoint models single GPU - complete
- resume training single GPU - complete 
- checkpoint models multi GPU - complete
- resume training multi GPU - complete
- remove old checkpoint files - complete
- add early stopping - complete
- sync loss across GPUs - TODO
- add wandb logging
"""
import copy
import time
import logging
import random
import os
import json

import pandas as pd
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import torch.distributed as dist
import torchmetrics
from tqdm import tqdm
import wandb

from .utils import get_optimizer, dist_log


class Trainer():

    def __init__(self,
                 args,
                 model,
                 train_loader,
                 val_loader=None,
                 test_loader=None):
        self.args = args
        self.model = model
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
        self.configure_optimizer()
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.early_stopping_counter = self.args.early_stopping_patience

        # if resume training and checkpoint exists, load it
        # get latest checkpoint
        checkpoint = None
        if args.resume:
            checkpoint = self._get_latest_checkpoint()
            if checkpoint is not None:
                dist_log(self.args, f'Resuming from checkpoint: {checkpoint}')
                self.load(checkpoint)
        if not args.resume or checkpoint is None:
            # TODO: move this to somewhere more visible
            if args.distributed:
                self.model.to(self.device)
                self.model = DDP(self.model,
                                 device_ids=[args.local_rank],
                                 output_device=args.local_rank)
            else:
                self.model.to(self.device)

        # distributed training attributes
        self.rank_0 = True
        if self.args.distributed:
            self.rank_0 = self.args.local_rank == 0
        # variable to determine if we're rank_0 in distributed mode or not in distributed mode at all
        self.rank_0_or_not_distributed = self.rank_0 or not self.args.distributed

        # wandb logging (will resume from checkpoint if exists)
        self.args.wandb = False
        if 'wandb_project' in args and args.wandb_project is not None:
            if (self.args.distributed and self.rank_0) or not self.args.distributed:
                self._init_wandb(checkpoint)
            

    def _init_wandb(self, checkpoint):
        self.args.wandb = True
        # if we're resuming a run, specify the run_id loaded from the checkpoint
        if self.args.resume and checkpoint is not None:
            self.wandb_run = wandb.init(project=self.args.wandb_project,
                                        config=self.args,
                                        id=self.wandb_run_id, # loaded from checkpoint
                                        resume='must')
            # TODO: do we want to relog the args?
            # wandb.log({'args': self.args})
        else:
            self.wandb_run = wandb.init(project=self.args.wandb_project,
                                        config=self.args)
            self.wandb_run_id = self.wandb_run.id
            # serialize device string
            args_copy = copy.deepcopy(self.args)
            args_copy.device = str(args_copy.device)
            wandb.log({'args': vars(args_copy)})
    
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

    def log(self, log_dict, step=None):
        """Logs a message to the wandb."""
        if not self.args.wandb:
            return
        if (self.args.distributed and self.rank_0) or not self.args.distributed:
            wandb.log(log_dict, step=step)
            

    def _update_metrics(self,
                        split,
                        num_correct,
                        num_examples,
                        total_loss,
                        data_time=None,
                        batch_time=None):
        """Updates the batch level metrics."""
        batch_metrics = self._batch_metrics[split]
        batch_metrics['num_correct'] += num_correct
        batch_metrics['num_examples'] += num_examples
        batch_metrics['total_loss'] += total_loss
        # batch_metrics['data_times'].append(data_time)
        # batch_metrics['batch_times'].append(batch_time)

    def _remove_old_checkpoints(self, best_checkpoint):
        """Removes old checkpoints."""
        # get all checkpoint directories
        checkpoints = [
            f for f in os.listdir(self.args.model_dir) if 'checkpoint' in f
            and os.path.isdir(os.path.join(self.args.model_dir, f))
        ]
        # sort checkpoints by step number (descending)
        checkpoints = sorted(checkpoints,
                             key=lambda x: int(x.split('-')[1]),
                             reverse=True)
        checkpoints = [
            os.path.join(self.args.model_dir, checkpoint)
            for checkpoint in checkpoints
        ]

        # remove the best checkpoint from the list (so it doesn't get deleted)
        checkpoints.remove(best_checkpoint)

        # remove all checkpoints from old runs (i.e. greater than the current global step)
        final_checkpoints = []
        for checkpoint in checkpoints:
            checkpoint_number = int(os.path.basename(checkpoint).split('-')[1])
            if checkpoint_number > self.global_step:
                dist_log(self.args, f'Removing old checkpoint: {checkpoint}')
                os.system(f'rm -rf {checkpoint}')
            else:
                final_checkpoints.append(checkpoint)

        # remove all extra checkpoints (those that are less than the current global step)
        keep_idx = self.args.num_checkpoints - 1
        if keep_idx >= 0 and keep_idx < len(final_checkpoints):
            for checkpoint in final_checkpoints[keep_idx:]:
                checkpoint_number = int(
                    os.path.basename(checkpoint).split('-')[1])
                if checkpoint_number < self.global_step:
                    dist_log(self.args,
                             f'Removing old checkpoint: {checkpoint}')
                    os.system(f'rm -rf {checkpoint}')

    def save(self, best=False):
        """Saves the model to disk."""
        # in distributed mode, only save from rank 0
        if self.args.distributed:
            if self.args.local_rank != 0:
                return

        save_dir = os.path.join(self.args.model_dir,
                                f'checkpoint-{self.global_step}')
        # if the directory exists, first remove it to avoid cross-contaminating with old runs
        if os.path.exists(save_dir):
            os.system(f'rm -rf {save_dir}')
        os.makedirs(save_dir)

        logging.info(f'Saving model to {save_dir}...')
        if self.args.distributed:
            model_obj = self.model.module.state_dict()
        else:
            model_obj = self.model.state_dict()
        torch.save(model_obj, os.path.join(save_dir, 'model.pt'))
        # save optimizer and lr scheduler state
        torch.save(self.optimizer.state_dict(),
                   os.path.join(save_dir, 'optimizer.pt'))
        torch.save(self.lr_scheduler.state_dict(),
                   os.path.join(save_dir, 'lr_scheduler.pt'))

        # save HF configs
        # config_file = os.path.join(save_dir, 'config.json')
        # with open(config_file, 'w') as f:
        #     json.dump(vars(self.model.config), f)

        # save args
        args = copy.deepcopy(self.args)

        # remove attributes that can't be serialized
        del args.device

        # if logging with wandb
        wandb_run_id = None
        if 'wandb_project' in args and args.wandb_project is not None:
            wandb_run_id = self.wandb_run_id

        # record the trainer state
        trainer_state = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'metrics': self.metrics,
            'args': vars(args),
            'wandb_run_id': wandb_run_id
        }
        with open(os.path.join(save_dir, 'trainer_state.json'), 'w') as f:
            json.dump(trainer_state, f)

        # if best == True, write a text file to args.model_dir to denote which checkpoint has the best val loss
        if best:
            with open(os.path.join(self.args.model_dir, 'best_checkpoint.txt'),
                      'w') as f:
                f.write(save_dir)

        # remove old checkpoints
        with open(os.path.join(self.args.model_dir, 'best_checkpoint.txt'),
                  'r') as f:
            best_checkpoint = f.read()
        self._remove_old_checkpoints(best_checkpoint=best_checkpoint)

    def _get_latest_checkpoint(self):
        # get the last checkpoint
        checkpoints = [
            f for f in os.listdir(self.args.model_dir) if 'checkpoint' in f
            and os.path.isdir(os.path.join(self.args.model_dir, f))
        ]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[1]))
        checkpoint = None
        if len(checkpoints) > 0:
            checkpoint = checkpoints[-1]
        return checkpoint

    def load(self, checkpoint=None):
        """Loads the model from disk."""
        # if checkpoint is None, load best based on best_checkpoint.txt
        if checkpoint is None:
            with open(os.path.join(self.args.model_dir, 'best_checkpoint.txt'),
                      'r') as f:
                checkpoint = f.read()
        elif checkpoint == 'last':
            checkpoint = self._get_latest_checkpoint()
            checkpoint = os.path.join(self.args.model_dir, checkpoint)
        else:
            checkpoint = os.path.join(self.args.model_dir, checkpoint)

        # load model, optimizer, and lr_scheduler
        save_dir = checkpoint

        # load trainer state
        # TODO: determine if this is what we want to do wrt the global step and running metrics
        with open(os.path.join(save_dir, 'trainer_state.json'), 'r') as f:
            trainer_state = json.load(f)
            self.global_step = trainer_state['global_step']
            self.epoch = trainer_state['epoch']
            self.metrics = trainer_state['metrics']
            # self.args = argparse.Namespace(**trainer_state['args'])
            self.wandb_run_id = trainer_state['wandb_run_id']

        # define device map so we load on rank 0 and broadcast to other ranks
        # https://discuss.pytorch.org/t/checkpoint-in-multi-gpu/97852/11
        map_location = None
        if self.args.distributed:
            map_location = f'cuda:{self.args.local_rank}'
            self.model.load_state_dict(
                torch.load(os.path.join(save_dir, 'model.pt'),
                           map_location=map_location))
            self.model.to(self.args.device)
            logging.info(
                f'Model device {self.model.device} on rank {self.args.local_rank}'
            )
            self.model = DDP(
                self.model,
                device_ids=[self.args.device],
                output_device=self.args.device,
            )
            # dist.barrier()
        else:
            self.model.load_state_dict(
                torch.load(os.path.join(save_dir, 'model.pt'),
                           map_location=map_location))

        logging.info(f'Loaded model on {self.args.device}...')
        self.optimizer.load_state_dict(
            torch.load(os.path.join(save_dir, 'optimizer.pt'),
                       map_location=map_location))
        logging.info(f'Loaded optimizer on {self.args.device}...')
        self.lr_scheduler.load_state_dict(
            torch.load(os.path.join(save_dir, 'lr_scheduler.pt'),
                       map_location=map_location))

    def train_step(self, batch):
        """Runs a single training step."""
        self.model.train()
        logging.debug(f'About to zero grad on {self.args.local_rank}')
        self.optimizer.zero_grad()
        logging.debug(
            f'About to put batch on GPU on rank {self.args.local_rank}')
        # move data to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        logging.debug(f'Batch on GPU on rank {self.args.local_rank}')
        # TODO: implement multi-task loss
        outputs = self.model(**batch)
        logging.debug(f'Forward pass on rank {self.args.local_rank}')
        loss = outputs[0]
        triplet_loss = None
        reporting_dict = {'train_loss': loss.item()}
        if self.args.triplet_loss:
            triplet_loss = outputs[2]
            if self.global_step % self.args.reporting_steps == 0:
                dist_log(self.args, f'Triplet loss: {triplet_loss.item():.2f} at step {self.global_step}')
            total_loss = loss + triplet_loss
            total_loss.backward()
            reporting_dict['triplet_loss'] = triplet_loss.item()
            reporting_dict['total_loss'] = total_loss.item()
        else:
            loss.backward()
        # log metrics
        self.log(reporting_dict, step=self.global_step,)
        logging.debug(f'Backward pass on rank {self.args.local_rank}')
        self.optimizer.step()
        self.lr_scheduler.step()
        self.log({'learning_rate': self.lr_scheduler.get_last_lr()[0]}, step=self.global_step)
        logging.debug(f'Optimization step on rank {self.args.local_rank}')

        preds = outputs[1].argmax(dim=1)
        labels = batch['labels']
        # compute accuracy
        # TODO: determine if this is forcing a sync and slowing down the pipeline
        num_correct = (preds == labels).sum().detach().item()
        logging.debug(f'Accuracy computation on rank {self.args.local_rank}')
        num_examples = len(preds)
        batch_accuracy = (num_correct / len(preds))
        loss = loss.detach().item()
        total_loss = loss * len(preds)
        self._update_metrics('train', num_correct, num_examples, total_loss)
        logging.debug(f'Updated metrics on rank {self.args.local_rank}')
        self._report(split='train')
        logging.debug(f'Reported metrics on rank {self.args.local_rank}')
        return loss, batch_accuracy

    def val_step(self, batch):
        """Runs a single evaluation step."""
        self.model.eval()
        # move data to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            if self.args.distributed:
                # need to call module to prevent hanging
                # https://discuss.pytorch.org/t/distributeddataparallel-barrier-doesnt-work-as-expected-during-evaluation/99867/7
                outputs = self.model.module(**batch)
            else:
                outputs = self.model(**batch)
            loss = outputs[0]
            preds = outputs[1].argmax(dim=1)
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
            loss = self._batch_metrics[split][
                'total_loss'] / self._batch_metrics[split]['num_examples']
            accuracy = self._batch_metrics[split][
                'num_correct'] / self._batch_metrics[split]['num_examples']
            dist_log(
                self.args,
                f'{split} average batch metrics at (epoch:batch) ({self.epoch}:{self.global_step}) - Loss {loss:.2f} - Accuracy - {accuracy:.2f}'
            )
            # report to wandb
            self.log(
                    {
                        f'{split}_loss': loss,
                        f'{split}_accuracy': accuracy
                    },
                    step=self.global_step)

    def _record_metrics(self, split='train'):
        # self.metrics[split]['epoch_data_times'].append(self._epoch_data_times)
        # self.metrics[split]['epoch_batch_times'].append(
        #     self._epoch_batch_times)
        # going to ignore batch level loss information in favor over epoch losses
        # self.metrics[split]['epoch_loss_vals'].append(self._epoch_loss_vals)
        epoch_loss = self._batch_metrics[split][
            'total_loss'] / self._batch_metrics[split]['num_examples']
        epoch_accuracy = self._batch_metrics[split][
            'num_correct'] / self._batch_metrics[split]['num_examples']
        self.metrics[split]['epoch_loss_vals'].append(epoch_loss)
        self.metrics[split]['epoch_accuracies'].append(epoch_accuracy)
        
        dist_log(self.args, f'{split} epoch metrics at (epoch:batch) ({self.epoch}:{self.global_step}) - Loss {epoch_loss:.2f} - Accuracy - {epoch_accuracy:.2f}')
        # TODO: determine why this isn't logging
        self.log(
                {
                    f'{split}_epoch_loss': epoch_loss,
                    f'{split}_epoch_accuracy': epoch_accuracy
                },
                step=self.global_step)

    def configure_optimizer(self):
        """Creates an optimizer (and potentially a learning rate scheduler) for the model."""
        self.optimizer, self.lr_scheduler = get_optimizer(
            self.args, self.model, **self.args.optimizer_kwargs)

    def train(self):
        # allow for variable number of batches per epoch
        if self.args.distributed:
            # https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html?highlight=join#torch.nn.parallel.DistributedDataParallel.join
            # throws an exception if any process terminates early, which is useful for maintaining consistent state
            # and early stopping
            # TODO: determine if the replicas are still in sync during training
            # particularly the learning rate scheduler, and anything else that might be a function of the global step/epoch
            try:
                with self.model.join(throw_on_early_termination=True):
                    self._train()
            except RuntimeError as e:
                # if uneven number of batches or early stopping, terminate training
                logging.debug(
                    f'Exception thrown on rank {self.args.local_rank}')
                logging.debug(e)
                logging.debug(
                    f'Terminating training on {self.args.local_rank}')
                return None
        else:
            self._train()

    def _train(self):
        """Runs training and evaluation for the specified number of epochs."""
        for epoch in range(self.epoch, self.epochs):
            self.epoch = epoch
            # set the epoch for the sampler so it shuffles appropriately
            # https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
            if self.args.distributed:
                self.train_loader.sampler.set_epoch(epoch)
                logging.debug(
                    f'Setting epoch to {str(self.train_loader.sampler.epoch)} on rank {str(self.args.local_rank)}'
                )

            # reset epoch level metrics
            self._reset_metrics('train')
            logging.debug(
                f'Length of train loader {len(self.train_loader)} on rank {self.args.local_rank}'
            )
            for idx, batch in tqdm(enumerate(self.train_loader),
                                   disable=not self.rank_0,
                                   total=len(self.train_loader),
                                   desc=f'Train Epoch {epoch}'):
                logging.debug(
                    f'Starting training step on rank {self.args.local_rank}')
                self.train_step(batch)
                logging.debug(
                    f'Finished training step {idx} on rank {self.args.local_rank}'
                )

                # potentially evaluate on validation set
                if self.args.val_steps > 0 and self.global_step % self.args.val_steps == 0:
                    # only run evaluation on rank 0 if using distributed training
                    if (self.args.distributed and self.args.local_rank
                            == 0) or not self.args.distributed:
                        logging.debug(
                            f'Running evaluation on validation set on rank {self.args.local_rank}'
                        )
                        self._reset_metrics(split='val')
                        for val_idx, val_batch in tqdm(
                                enumerate(self.val_loader),
                                disable=not self.rank_0,
                                total=len(self.val_loader),
                                desc=f'Validation Epoch {epoch}'):
                            self.val_step(val_batch)
                            logging.debug(
                                f'Finished validation step {val_idx} on rank {self.args.local_rank}'
                            )
                        self._report(split='val')
                        self._record_metrics(split='val')

                    # save model checkpoint
                    if self.args.num_checkpoints > 0:
                        # TODO: refactor this into a separate function (too much nesting)
                        # only run on rank 0 if using distributed training
                        if (self.args.distributed and self.rank_0) or not self.args.distributed:
                            # TODO: sync loss across ranks before deciding to save
                            new_best = False
                            # on first iteration [-1] might out of range
                            current_loss = self.best_val_loss
                            if len(self.metrics['val']['epoch_loss_vals']) >= 1:
                                current_loss = self.metrics['val'][
                                    'epoch_loss_vals'][-1]
                            if self.best_val_loss > current_loss:
                                self.best_val_loss = current_loss
                                logging.info(
                                    f'New best loss of {self.best_val_loss:.2f}. Saving model on rank {self.args.local_rank}.'
                                )
                                new_best = True
                                # reset early stopping counter
                                self.early_stopping_counter = self.args.early_stopping_patience
                            else:
                                self.early_stopping_counter -= 1
                                logging.info(
                                    f'No improvement in validation loss. Early stopping counter at {self.early_stopping_counter}'
                                )
                            self.save(best=new_best)
                            # returning will send signal to other ranks to stop training
                            if self.early_stopping_counter == 0:
                                logging.info(
                                    f'Early stopping counter at 0. Stopping training on rank {self.args.local_rank}'
                                )
                                return None

                # # ensure all ranks are synced before moving on to next batch
                # if self.args.distributed:
                #     dist.barrier()
                #     logging.debug(
                #         f'Rank {self.args.local_rank} finished training step {epoch}:{idx} (global step: {self.global_step}) and barrier sync'
                #     )

                self.global_step += 1

            # record metrics
            self._record_metrics(split='train')
            # if self.args.distributed:
            #     dist.barrier()
        logging.debug(
            f'Returning from train method on rank {self.args.local_rank}')
        return None

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