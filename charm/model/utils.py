from functools import partial
import os

from datasets import load_dataset
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.utils.data import Dataset
import torch
from transformers import get_linear_schedule_with_warmup, DataCollatorWithPadding, AutoTokenizer


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


def get_circumpex_dataset(tokenizer, data_dir, debug_pct):
    # load the dataset
    df = pd.read_csv(os.path.join(data_dir,
                                  'gpt_labels_r1_mini_eval_text.csv'))
    label2id = {
        'Assured-Dominant': 0,
        'Gregarious-Extraverted': 1,
        'Warm-Agreeable': 2,
        'Unassuming-Ingenuous': 3,
        'Unassured-Submissive': 4,
        'Aloof-Introverted': 5,
        'Cold': 6,
        'Arrogant-Calculating': 7
    }
    id2label = {v: k for k, v in label2id.items()}

    # add a label column
    df['label'] = df['social_orientation'].map(label2id)

    # split intro train and val set by filename
    train_percent = 0.8
    filenames = df['filename'].unique()
    # shuffle filenames
    np.random.shuffle(filenames)
    train_filenames = filenames[:int(train_percent * len(filenames))]
    val_filenames = filenames[int(train_percent * len(filenames)):]

    # if debug_pct is set, use a subset of the data
    train_filenames = train_filenames[:int(debug_pct * len(train_filenames))]
    val_filenames = val_filenames[:int(debug_pct * len(val_filenames))]

    # create the datasets
    train_dataset = CircumplexDataset(df[df['filename'].isin(train_filenames)],
                                      tokenizer,
                                      window_size=8,
                                      stride=1)
    val_dataset = CircumplexDataset(df[df['filename'].isin(val_filenames)],
                                    tokenizer,
                                    window_size=8,
                                    stride=1)
    return train_dataset, val_dataset, id2label, label2id


def preprocess_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True)


def sample_dataset(dataset, sample_pct):
    """Sample from the dataset for faster debugging."""
    train_sample_size = int(len(dataset['train']) * sample_pct)
    test_sample_size = int(len(dataset['test']) * sample_pct)
    dataset['train'] = dataset['train'].shuffle(seed=42).select(
        range(train_sample_size))
    dataset['test'] = dataset['test'].shuffle(seed=42).select(
        range(test_sample_size))
    return dataset


def get_imdb_dataset(tokenizer, debug_pct, remove_unused_columns=True):
    imdb = load_dataset("imdb", split={'train': 'train', 'test': 'test'})
    imdb = sample_dataset(imdb, debug_pct)

    preprocess_function_partial = partial(preprocess_function,
                                          tokenizer=tokenizer)

    tokenized_imdb = imdb.map(preprocess_function_partial, batched=True)
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}
    if remove_unused_columns:
        tokenized_imdb = tokenized_imdb.remove_columns(
            ['text', 'attention_mask'])
    return tokenized_imdb['train'], tokenized_imdb['test'], id2label, label2id


def get_dataloaders(config,
                    collate_fn,
                    train_dataset,
                    val_dataset,
                    test_dataset=None):
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn)
    if test_dataset is not None:
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.batch_size * 2,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_fn)
        return train_dataloader, val_dataloader, test_dataloader
    return train_dataloader, val_dataloader


def get_optimizer(args, model, **kwargs):
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            **kwargs,
        )
    elif args.optimizer == 'SGD-Nesterov':
        optimizer = optim.SGD(
            model.parameters(),
            **kwargs,
        )
    elif args.optimizer == 'Adagrad':
        optimizer = optim.Adagrad(
            model.parameters(),
            **kwargs,
        )
    elif args.optimizer == 'Adadelta':
        optimizer = optim.Adadelta(
            model.parameters(),
            **kwargs,
        )
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            **kwargs,
        )
    elif args.optimizer == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(),
            **kwargs,
        )
    
    # get learning rate scheduler
    scheduler = None
    if args.lr_scheduler == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.num_train_steps,
        )
    return optimizer, scheduler

def get_data(args):
    # get the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    # get the dataset
    if args.fast_dev_run and args.dataset != 'circumplex':
        # useful for sanity checking the system
        data = get_imdb_dataset(tokenizer, debug_pct=0.1)
    else:
        # get the Circumplex dataset
        data = get_circumpex_dataset(tokenizer, args.data_dir,
                                     args.debug_pct)

    train_dataset, val_dataset, id2label, label2id = data
    args.id2label = id2label
    args.label2id = label2id
    collate = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataloader, val_dataloader = get_dataloaders(args, collate,
                                                       train_dataset,
                                                       val_dataset)
    return train_dataloader, val_dataloader, args