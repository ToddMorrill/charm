import logging
from functools import partial
import os
import random

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


def get_dataloaders(args,
                    collate_fn,
                    train_dataset,
                    val_dataset,
                    test_dataset=None,
                    train_collate_fn=None):
    train_sampler = None
    if args.distributed:
        # only use distributed sampler for training data because this sampler
        # pads the dataset to be divisible by the number of processes
        # which will mess up the validation and test data
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, 
            num_replicas=args.world_size,
            rank=args.local_rank,
            shuffle=True)
    train_collate_fn = collate_fn if train_collate_fn is None else train_collate_fn
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=1,
        pin_memory=True,
        collate_fn=train_collate_fn,
        sampler=train_sampler)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn)
    if test_dataset is not None:
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size * 2,
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
    if args.lr_scheduler == 'linear-with-warmup':
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.num_train_steps,
        )
    return optimizer, scheduler

def get_triples(labels, num_classes):
    """Generates indexes corresponding to positive and negative examples for
    each example in labels."""
    # for each class, get the index of the examples with that class
    class_indexes = {}
    for i in range(num_classes):
        class_examples = torch.where(labels == i)[0]
        if len(class_examples) > 0:
            class_indexes[i] = class_examples

    # treat each label in labels as an anchor
    # for each label in labels, generate a positive and negative example
    anchors = []
    positives = []
    negatives = []
    for anchor_idx in range(len(labels)):
        # get the anchor class
        anchor_class = labels[anchor_idx].item()

        # if no positive examples to compare to or if no negative classes present, continue
        if len(class_indexes[anchor_class]) == 1 or len(class_indexes) == 1:
            continue

        # select a random positive example with the same class label from class_indexes
        positive_idx = anchor_idx
        while positive_idx == anchor_idx:
            positive_idx = random.choice(class_indexes[anchor_class])

        # randomly select a negative class
        negative_class = anchor_class
        while negative_class == anchor_class:
            negative_class = random.choice(list(class_indexes.keys()))

        # randomly select a negative example from the negative class
        negative_idx = random.choice(class_indexes[negative_class])

        # add the anchor, positive, and negative examples to the lists
        anchors.append(anchor_idx)
        positives.append(positive_idx)
        negatives.append(negative_idx)
    # convert to tensors
    anchors = torch.tensor(anchors, dtype=torch.long)
    positives = torch.tensor(positives, dtype=torch.long)
    negatives = torch.tensor(negatives, dtype=torch.long)
    return anchors, positives, negatives

def triplet_collate(tokenizer, num_labels):
    padding_collate = DataCollatorWithPadding(tokenizer=tokenizer)
    def collate(batch):
        # pad sequences and get attention masks
        batch = padding_collate(batch)
        # get the labels
        labels = batch['labels']

        # get the anchor, positive, and negative examples
        anchors, positives, negatives = get_triples(labels, num_labels)
        batch['anchors'] = anchors
        batch['positives'] = positives
        batch['negatives'] = negatives
        return batch
    return collate

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
    train_collate = None
    if args.triplet_loss:
        train_collate = triplet_collate(tokenizer, len(id2label))
    collate = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataloader, val_dataloader = get_dataloaders(args, collate,
                                                       train_dataset,
                                                       val_dataset, train_collate_fn=train_collate)
    return train_dataloader, val_dataloader, args

def dist_log(args, message):
    if args.distributed and args.local_rank == 0:
        logging.info(message)
    elif not args.distributed:
        logging.info(message)