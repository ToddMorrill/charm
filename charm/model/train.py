"""This module is responsible for training the communication change model.

Examples:
    $ python -u -m charm.model.train \
        --data-dir ~/Documents/data/charm/transformed/LDC2022E11_CCU_TA1_Mandarin_Chinese_Development_Source_Data_R1/data/text \
        --model-dir ~/Documents/data/charm/models/xlm-roberta-base
"""
import argparse
import logging
import os
import warnings
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
from evaluate import load
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
from torch.optim import SGD
from transformers import EarlyStoppingCallback
from tqdm import tqdm
import transformers

transformers.logging.set_verbosity_warning()
pd.options.mode.chained_assignment = None

class CircumplexDataset(Dataset):
    """Pretokenizes the text and combines window size utterances into one
    sample, adding special tokens, as needed, when generating the example.
    """
    def __init__(self, df, tokenizer, window_size=5, stride=1):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = tokenizer.model_max_length
        self.window_size = window_size
        self.stride = stride

        # number of filenames
        self.num_filenames = len(self.df['filename'].unique())
        # number of examples that can be generated per filename
        # for each filename, this is the number of rows minus the window size
        # divided by the stride, plus one
        # [(rows - window_size) / stride] + 1
        # TODO: implement some sort padding scheme to handle stride > 1
        self.num_examples_per_filename = (self.df.groupby('filename').size() -
                                          self.window_size) // self.stride + 1
        self.total_examples = self.num_examples_per_filename.sum()

        # pretokenize the text
        # TODO: move over to an apache beam pipeline
        # though there's not really an easy way to do this without replicating
        # the data many times
        self.df['input_ids'] = self.tokenizer(
            self.df['ORIGINAL_TEXT'].values.tolist(),
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            return_attention_mask=False)['input_ids']

    def __len__(self):
        # length is the number of examples that can be generated per filename
        # times the number of filenames
        return len(self.df)

    def __getitem__(self, index):
        row = self.df[['input_ids', 'label']].iloc[index]
        # TODO: concatenate window_size utterances into one sample
        # TODO: the commented usage is for multiple rows (i.e. an iloc slice)
        # return row.to_dict(orient='list')
        return row.to_dict()


def compute_metrics(eval_pred):
    acc = load('accuracy')
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return acc.compute(predictions=predictions, references=labels)


def early_stopping(dataset, per_device_train_batch_size, num_devices, eval_steps):
    # set patience to be roughly 1 epoch
    iters_per_epoch = len(dataset) // (
        per_device_train_batch_size * num_devices)
    num_evals_per_epoch = iters_per_epoch // eval_steps
    patience = num_evals_per_epoch * 10
    early_stop = EarlyStoppingCallback(
        early_stopping_patience=patience)
    return early_stop

def main(args):
    np.random.seed(42)
    torch.manual_seed(42)

    # load the dataset
    data_dir = args.data_dir
    df = pd.read_csv(os.path.join(data_dir, 'text_circumplex_random.csv'))
    # create a label column based on the unique values in the social_orientation column
    df['label'], label_mapping = pd.factorize(df['social_orientation'])

    model = 'xlm-roberta-base'
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model, num_labels=len(label_mapping))
    
    # split intro train and val set by filename
    train_percent = 0.8
    filenames = df['filename'].unique()
    # shuffle filenames
    np.random.shuffle(filenames)
    train_filenames = filenames[:int(train_percent * len(filenames))]
    val_filenames = filenames[int(train_percent * len(filenames)):]
    debug_pct = 0.1
    train_filenames = train_filenames[:int(debug_pct * len(train_filenames))]
    val_filenames = val_filenames[:int(debug_pct * len(val_filenames))]
    train_dataset = CircumplexDataset(df[df['filename'].isin(train_filenames)],
                                      tokenizer,
                                      window_size=8,
                                      stride=1)
    val_dataset = CircumplexDataset(df[df['filename'].isin(val_filenames)],
                                    tokenizer,
                                    window_size=8,
                                    stride=1)
    collate = DataCollatorWithPadding(tokenizer=tokenizer)
    # inputs = collate(train_dataset[:10])

    train_batch_size = 32
    eval_log_steps = 10
    training_args = TrainingArguments(
            output_dir=args.model_dir,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=train_batch_size*2,
            gradient_accumulation_steps=1,
            # learning_rate=lr,
            # TODO: understand this parameter
            overwrite_output_dir=True,
            evaluation_strategy='steps',
            eval_steps=eval_log_steps,
            logging_steps=eval_log_steps,
            dataloader_num_workers=0,
            # num_epochs determined time average time to converge in experiments
            num_train_epochs=10,
            save_total_limit=2,
            load_best_model_at_end=True,
            save_strategy='steps',
            save_steps=eval_log_steps,
            seed=42)

    early_stop = early_stopping(train_dataset, train_batch_size, 3, eval_log_steps)
    # early_stop = None
    callbacks = [early_stop] if early_stop else None
    trainer = Trainer(model,
                               training_args,
                               train_dataset=train_dataset,
                               eval_dataset=val_dataset,
                               data_collator=collate,
                               tokenizer=tokenizer,
                               compute_metrics=compute_metrics,
                               callbacks=callbacks,
                               )

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',
                        type=str,
                        help='Directory where the raw data data is stored.')
    parser.add_argument('--model-dir',
                        type=str,
                        help='Directory where the model will be stored.')
    args = parser.parse_args()
    main(args)