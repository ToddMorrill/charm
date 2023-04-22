"""This script takes a model path as input, loads the model, and makes
predictions on the val and test sets. It then score the validation predictions
against the NIST scoring tool, selects a filtering procedure, and saves the
predictions to disk.

Examples:
    $ CUDA_VISIBLE_DEVICES=2,3 nohup python -m charm.model.predict \
        --model-paths /mnt/swordfish-pool2/ccu/models/change-point-medium-reweight \
        --output-dir /mnt/swordfish-pool2/ccu/predictions \
        --device 0 \
        --final-eval \
        --transcript whisper \
        --parallel \
        --batch-size 512 \
        --num-workers 1 \
        > eval_whisper.log 2>&1 &
    
    $ CUDA_VISIBLE_DEVICES=0,1,2,3 python -m charm.model.predict \
        --model-paths /mnt/swordfish-pool2/ccu/models/change-point-medium-reweight \
        --output-dir /mnt/swordfish-pool2/ccu/predictions \
        --final-eval \
        --transcript azure \
        --parallel \
        --batch-size 512 \
        --num-workers 2 \
        > eval_azure.log 2>&1 &
    
    $ CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m charm.model.predict \
        --model-paths /mnt/swordfish-pool2/ccu/models/change-point-medium-reweight \
        --output-dir /mnt/swordfish-pool2/ccu/predictions \
        --final-eval \
        --transcript wav2vec \
        --parallel \
        --batch-size 512 \
        --num-workers 1 \
        > eval_wav2vec.log 2>&1 &

TODO:
- add ability to predict social orientation
- distributed training support
"""
import os
import json
import logging
import argparse
import time
from functools import cache
import statistics

import numpy as np
import pandas as pd
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import wandb

from .model import XLMRClassificationPlusTripletLoss
from .utils import get_data, get_best_model, get_eval_dataloader, ChangePointDataset
from ..loaders.ldc_data import load_ldc_data
from .average_precision import calculate_average_precision, filter_system_preds

class Predictor(object):
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.load_config()
    
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
    
    def load_config(self, checkpoint=None):
        """Loads the model from disk."""
        # if checkpoint is None, load best based on best_checkpoint.txt
        if checkpoint is None:
            with open(os.path.join(self.model_dir, 'best_checkpoint.txt'),
                        'r') as f:
                checkpoint = f.read()
        elif checkpoint == 'last':
            checkpoint = self._get_latest_checkpoint()
            checkpoint = os.path.join(self.model_dir, checkpoint)
        else:
            checkpoint = os.path.join(self.model_dir, checkpoint)
        
        self.checkpoint = checkpoint

        # load trainer state
        with open(os.path.join(checkpoint, 'trainer_state.json'), 'r') as f:
            trainer_state = json.load(f)
            self.global_step = trainer_state['global_step']
            self.epoch = trainer_state['epoch']
            self.metrics = trainer_state['metrics']
            self.wandb_run_id = trainer_state['wandb_run_id']
            self.args = argparse.Namespace(**trainer_state['args'])
        
        
    def load_model(self, model):
        self.model = model
        # load model
        # define device map so we load on rank 0 and broadcast to other ranks
        # https://discuss.pytorch.org/t/checkpoint-in-multi-gpu/97852/11
        map_location = None
        # TODO: will need to adjust args to support this properly
        if self.args.distributed:
            map_location = f'cuda:{self.args.local_rank}'
            self.model.load_state_dict(
                torch.load(os.path.join(self.checkpoint, 'model.pt'),
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
                torch.load(os.path.join(self.checkpoint, 'model.pt'),
                            map_location=map_location))
            self.model.to(self.args.device)
        
        # put the model in eval mode
        self.model.eval()
        logging.info(f'Loaded model on {self.args.device}...')
    
    def predict(self, loader):
        # make model predictions
        with torch.no_grad():
            predictions = []
            llr = []
            labels = []
            for batch in tqdm(loader):
                if 'labels' in batch:
                    labels.extend(batch['labels'])
                # move data to device
                batch = {k: v.to(self.args.device) for k, v in batch.items()}

                outputs = self.model(**batch)
                logits = outputs[1]
                llr.extend((logits[:, 1] - logits[:, 0]).tolist())
                preds = torch.argmax(logits, dim=1)
                predictions.extend(preds.tolist())

        return predictions, llr, labels

def get_changepoints(df):
    # get the change points as follows
    # filter for utterances where the prediction is 1
    # get the midpoint of that utterance (or start or end?) as the timestamp
    change_point_df = df[df['prediction'] == 1]
    change_point_df = change_point_df[['file_id', 'start', 'data_type', 'llr']].rename(columns={'start': 'timestamp', 'data_type': 'type'})
    hyps = change_point_df.to_dict(orient='records')
    return hyps
    
def load_metadata():
    df = pd.read_csv(os.path.join(
        '/mnt/swordfish-pool2/ccu/transformed/change-point', 'change_point_social_orientation_train_val_test.csv'),
                     index_col=0)
    # split into train, val, and test sets
    train_df = df[df['split'] == 'train'].reset_index(drop=True).reset_index()
    val_df = df[df['split'] == 'val'].reset_index(drop=True).reset_index()
    test_df = df[df['split'] == 'test'].reset_index(drop=True).reset_index()
    train_df = train_df.set_index(['index', 'file_id'])
    val_df = val_df.set_index(['index', 'file_id'])
    test_df = test_df.set_index(['index', 'file_id'])
    return train_df, val_df, test_df

def load_reference_data(split='INTERNAL_VAL'):
    # load the reference data to be sure we've got all the data points
    data = load_ldc_data(False, True)
    val_data = {k: v for k, v in data.items() if split in data[k]['splits']}
    # test_data = {k: v for k, v in data.items() if data[k]['split'] == 'test'}
    refs = []
    for file_id in val_data:
        for changepoint in val_data[file_id]['changepoints']:
            refs.append({'file_id': file_id, 'timestamp': changepoint['timestamp'], 'type': val_data[file_id]['data_type'], 'impact_scalar': changepoint['impact_scalar']})
    return refs

def score_nist(refs, hyps):
    # filtering options {'none', 'highest', 'lowest', 'most_similar'}
    start = time.perf_counter()
    results = {}
    mean_scores = []
    for filtering in ['none', 'highest', 'lowest', 'most_similar']:
        logging.info(f'filtering with {filtering}')
        results[filtering] = calculate_average_precision(refs,
                                    hyps,
                                    text_char_threshold=100,
                                    time_sec_threshold=10,
                                    filtering=filtering)
        results[filtering]['harmonic_mean'] = statistics.harmonic_mean(results[filtering].values())
        mean_scores.append((filtering, statistics.harmonic_mean(results[filtering].values())))
    end = time.perf_counter()
    logging.info(f'Time taken to evaluation: {end - start:.2f} seconds')
    return results, mean_scores


def _run_eval(predictor, loader, df):
    # make predictions
    preds, llr, val_labels = predictor.predict(loader)
    preds_df = pd.DataFrame({'prediction': preds, 'llr': llr})

    # prep predictions, join back to df
    df = pd.concat((df.reset_index(), preds_df), axis=1)
    
    # get the change points
    hyps = get_changepoints(df)
    return hyps, df

def dump_preds_results(final_preds, df, predictor, output_dir, split='val', results=None):
    # dump predictions to csv with the following columns
    # file_id, timestamp, llr
    final_preds_df = pd.DataFrame.from_dict(final_preds)[['file_id', 'timestamp', 'llr']]

    # save to predictions folder with the same name as the model_dir
    preds_dir = os.path.join(output_dir, os.path.basename(predictor.model_dir))
    os.makedirs(preds_dir, exist_ok=True)
    final_preds_df.to_csv(os.path.join(preds_dir, f'{split}.csv'), index=False)

    # save detailed predictions linked to the metadata
    preds_filepath = os.path.join(preds_dir, f'{split}_preds_detailed.csv')
    df.to_csv(preds_filepath)

    # dump results to json
    if results is not None:
        results_filepath = os.path.join(preds_dir, f'{split}_results.json')
        with open(results_filepath, 'w') as f:
            json.dump(results, f)
    
        # log to wandb
        wandb.log({f'{split}_AP': results})


def run_eval(predictor, val_loader, test_loader, val_df, test_df, val_refs, output_dir):
    # make predictions
    logging.info('Making predictions on validation set...')
    val_hyps, val_df = _run_eval(predictor, val_loader, val_df)

    # score the predictions
    logging.info('Scoring predictions on validation set...')
    val_results, val_mean_scores = score_nist(val_refs, val_hyps)
    logging.info(f'Validation set results: {val_results}')

    # get the best filtering approach
    filtering = max(val_mean_scores, key=lambda x: x[1])[0]

    # TODO: move this inot dump_preds_results
    val_mean_scores = {k: v for k, v in val_mean_scores}
    wandb.log({'val_harmonic_mean_AP': val_mean_scores})

    # filter predictions according to the best procedure
    final_val_preds = filter_system_preds(val_hyps, text_char_threshold=100, time_sec_threshold=10, filtering=filtering)

    logging.info('Saving validation set results...')
    dump_preds_results(final_val_preds, val_df, predictor, output_dir, split='val', results=val_results)
    
    # make predictions on test set
    logging.info('Making predictions on test set...')
    test_hyps, test_df = _run_eval(predictor, test_loader, test_df)

    # filter predictions according to the best procedure
    final_test_preds = filter_system_preds(test_hyps, text_char_threshold=100, time_sec_threshold=10, filtering=filtering)

    logging.info('Saving test set results...')
    dump_preds_results(final_test_preds, test_df, predictor, output_dir, split='test', results=None)


def eval_pipeline(args):
    predictor = Predictor(args.model_path)
    if args.val:
        # "resume" the wandb run so we can track metrics
        wandb_run = wandb.init(project=predictor.args.wandb_project,
                                        id=predictor.wandb_run_id, # loaded from checkpoint
                                        resume='must')
    predictor.args.device = f'cuda:{args.device}'
    model = XLMRClassificationPlusTripletLoss.from_pretrained(
            predictor.args.model_name_or_path,
            num_labels=len(predictor.args.id2label),
            id2label=predictor.args.id2label,
            label2id=predictor.args.label2id)
    # this will enable things like triplet loss, impact scalar, social orientation, and class weighting
    model.add_args(predictor.args)
    predictor.load_model(model)

    if args.parallel:
        # TODO: make these command line args
        predictor.model = torch.nn.DataParallel(predictor.model)

    if args.val or args.test:
        # load data
        train_loader, val_loader, test_loader, predictor.args = get_data(predictor.args)
        #load metadata
        train_df, val_df, test_df = load_metadata()

        # load val reference data
        val_refs = load_reference_data()

        # run eval
        run_eval(predictor, val_loader, test_loader, val_df, test_df, val_refs, args.output_dir)

        # finish wandb run
        wandb.finish()
        return
    
    # load eval data
    if args.final_eval:
        logging.info('Running on final eval set...')
        # sample to debug
        eval_df = prepare_eval_data(args)
        # create ChangePointDataset
        eval_dataset = ChangePointDataset(eval_df, AutoTokenizer.from_pretrained(predictor.args.model_name_or_path), predictor.args.window_size, predictor.args.impact_scalar, predictor.args.social_orientation)
        # dataloader
        num_gpus = torch.cuda.device_count()
        logging.info(f'Running on {num_gpus} GPUs with a batch size of {args.batch_size * num_gpus} and {args.num_workers*num_gpus} workers.')
        eval_loader = get_eval_dataloader(batch_size=args.batch_size * num_gpus, num_workers=args.num_workers*num_gpus, dataset=eval_dataset)
        batch = next(iter(eval_loader))
        hyps, df = _run_eval(predictor, eval_loader, eval_df)
        # filter predictions according to the best procedure
        # TODO: generalize this to use the best filtering approach
        final_eval_preds = filter_system_preds(hyps, text_char_threshold=100, time_sec_threshold=10, filtering='lowest')
        logging.info('Saving final eval set results...')
        dump_preds_results(final_eval_preds, df, predictor, args.output_dir, split=f'eval_{args.transcript}', results=None)

def prepare_eval_data(args, cache=True):
    # TODO: generalize this to work for any split
    filepath = os.path.join(args.data_dir, f'LDC2023E07_{args.transcript}.csv')
    if cache and os.path.exists(filepath):
        logging.info(f'Loading cached data from {filepath}')
        cache_df = pd.read_csv(filepath, index_col=0)
        # reestablish the index
        cache_df = cache_df.reset_index(drop=True).reset_index()
        cache_df = cache_df.set_index(['index', 'file_id'])
        return cache_df
    
    # otherwise load and process the data
    data = load_ldc_data(False, True)
    transcript = args.transcript
    eval_data = {}
    for file_id in data:
        if 'EVALUATION_LDC2023E07' in data[file_id]['splits'] and data[file_id]['processed']:
            eval_data[file_id] = data[file_id]
            # only keep the transcript we want
            if data[file_id]['data_type'] in ['video', 'audio']:
                eval_data[file_id]['utterances'] = data[file_id]['utterances'][transcript]

    logging.info(f'Loaded {len(eval_data)} files for split EVALUATION_LDC2023E07')

    df = pd.DataFrame.from_dict(eval_data, orient='index')

    # convert to dataframe
    utterance_df = df.drop(columns=['changepoints'])
    utterance_df = utterance_df.explode('utterances')
    logging.info(f'{len(utterance_df):,} utterances in evaluation dataset total.')
    utterance_df = utterance_df.reset_index(drop=True)

    utterance_df = pd.concat((utterance_df, pd.json_normalize(utterance_df['utterances'])), axis=1)
    utterance_df = utterance_df.drop(columns=['utterances'])
    # the annotated segment is called start and end as well as the timestamps for individual utterances
    utterance_df.columns.values[2] = 'anno_start'
    utterance_df.columns.values[3] = 'anno_end'

    # set anno_start to be 0 and anno_end to be float('inf')
    utterance_df['anno_start'] = 0
    utterance_df['anno_end'] = float('inf')

    # sort values by file_id and start to be safe
    utterance_df = utterance_df.sort_values(by=['file_id', 'start'], ascending=True)
    utterance_df = utterance_df.reset_index(drop=True)
    # give ourselves an index to work with
    utterance_df = utterance_df.reset_index()
    # try to put an index on both the index and the file_id to speed up the slicing
    utterance_df = utterance_df.set_index(['index', 'file_id'])
    
    # save to cache
    if cache:
        logging.info(f'Saving cached data to {filepath}')
        utterance_df.to_csv(filepath, index=True)
    return utterance_df


def main(args):
    for model_path in args.model_paths:
        args.model_path = model_path
        eval_pipeline(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model-paths',
        type=os.path.expanduser,
        required=True,
        nargs='*',
        help='The path to the model to use for prediction.')
    parser.add_argument('--data-dir', type=os.path.expanduser, default='/mnt/swordfish-pool2/ccu/transformed/change-point', help='The path to the data directory.')
    parser.add_argument(
        '--output-dir',
        type=os.path.expanduser,
        default='/mnt/swordfish-pool2/ccu/predictions',
        help='The path to the output directory.')
    parser.add_argument('--device', type=int, default=0, help='The GPU device to use.')
    parser.add_argument('--val', action='store_true', default=False, help='Evaluate on the validation set.')
    parser.add_argument('--test', action='store_true', default=False, help='Evaluate on the test set. Note that this will also evaluate on the validation set to pick the best filtering method.')
    parser.add_argument('--final-eval', action='store_true', default=False, help='Evaluate on the final evaluation set. Note that this will also evaluate on the validation set to pick the best filtering method.')
    parser.add_argument('--transcript', required=True, help='The transcript to use for evaluation. Must be one of {whisper, azure, wav2vec}.')
    parser.add_argument('--parallel', action='store_true', default=False, help='Run evaluation in parallel on all available GPUs.')
    parser.add_argument('--num-workers', type=int, default=1, help='The number of workers to use for the dataloader per GPU device.')
    parser.add_argument('--batch-size', type=int, default=512, help='The batch size to use for evaluation per device.')
    args = parser.parse_args()
    logging.basicConfig(datefmt='%Y-%m-%d %I:%M:%S',
                        format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO)
    main(args)

