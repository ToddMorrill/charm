import logging
from functools import partial
import os
import random
import json
from functools import cache

from datasets import load_dataset
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.utils.data import Dataset
import torch
from transformers import get_linear_schedule_with_warmup, DataCollatorWithPadding, AutoTokenizer

from ..loaders.ldc_data import load_ldc_data


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


class ChangePointDataset(Dataset):
    """Pretokenizes the text and combines window size utterances into one
    sample, adding special tokens, as needed, when generating the example.
    """
    def __init__(self,
                 df,
                 tokenizer,
                 window_size=4,
                 impact_scalar=False,
                 social_orientation=False):
        self.df = df
        self.labeled = 'labels' in self.df.columns
        self.tokenizer = tokenizer
        self.max_len = tokenizer.model_max_length
        self.window_size = window_size
        self.impact_scalar = impact_scalar
        self.social_orientation = social_orientation

        # pretokenize the text
        # TODO: move over to an apache beam pipeline
        # though there's not really an easy way to do this without replicating
        # the data many times
        # TODO: add special tokens to the text
        # TODO: use ground truth social orientation labels
        if social_orientation:
            # e.g. [Arrogant-Calculating] 还是挣那么多钱
            self.df['text_final'] = self.df['social_orientation_preds'].apply(
                lambda x: f'[{x}]') + ' ' + self.df['text']
        else:
            self.df['text_final'] = self.df['text']
        
        # M01004G0B from the evaluation set is missing the text field, fill with empty string
        self.df['text_final'] = self.df['text_final'].fillna('')

        self.df['input_ids'] = self.tokenizer(
            self.df['text_final'].values.tolist(),
            add_special_tokens=False,
            max_length=self.max_len,
            truncation=True,
            return_attention_mask=False)['input_ids']

        # build a map from index to file_id
        self.idx_file_id_map = {idx: file_id for idx, file_id in self.df.index}
        self.file_id_start_end_df = {}
        for file_id in self.df.index.get_level_values('file_id').unique():
            file_df = self.df.xs(file_id, level=1, drop_level=False)
            first_idx = file_df.iloc[0].name[0]
            last_idx = file_df.iloc[-1].name[0]
            self.file_id_start_end_df[file_id] = (first_idx, last_idx, file_df)

    def _get_tokens(self, input_id_list):
        tokens = [self.tokenizer.cls_token_id]
        for idx, utterance in enumerate(input_id_list):
            # add a sep token between utterances
            if idx > 0:
                tokens.append(self.tokenizer.sep_token_id)
            tokens.extend(utterance)
            tokens.append(self.tokenizer.eos_token_id)

        # if the sequence is too long, truncate half from the beginning and half from the end
        # TODO: with this, you get the occasial sequence that starts with a sep token
        if len(tokens) > self.max_len:
            overage = len(tokens) - self.max_len
            tokens = tokens[((overage // 2) + 2):-((overage // 2) + 2)]
            # add the cls and eos tokens back
            tokens = [self.tokenizer.cls_token_id
                      ] + tokens + [self.tokenizer.eos_token_id]
        return tokens

    def __len__(self):
        # length is the number of examples that can be generated per filename
        # times the number of filenames
        return len(self.df)

    # @cache
    def __getitem__(self, idx):
        # TODO: speed this up somehow
        file_id = self.idx_file_id_map[idx]
        first_idx, last_idx, file_df = self.file_id_start_end_df[file_id]
        # turns out that using iloc is faster than using loc
        # translate the idx to a row number
        idx_row_num = idx - first_idx
        # get the start and end indices for the window
        start_idx = max(0, idx_row_num - self.window_size)
        end_idx = min(last_idx - first_idx,
                      idx_row_num + (self.window_size - 1))

        # start_idx = max(first_idx, idx - self.window_size)
        # end_idx = min(last_idx, idx + (self.window_size - 1))
        # # .loc[start_idx:end_idx] is inclusive
        # utterances = file_df.loc[start_idx:end_idx]
        utterances = file_df.iloc[start_idx:end_idx + 1]
        # print(f'utterances are {utterances}')
        input_id_list = utterances['input_ids'].values.tolist()
        tokens = self._get_tokens(input_id_list)
        # print(f'tokens is {tokens}')

        # if labels not present just return input_ids
        if not self.labeled:
            return {
                'input_ids': tokens,
            }
        
        # label should be the max label in the window (i.e. greedily label change points)
        # i.e. if any of the utterances in the window are change points, then the window is a change point
        label = utterances['labels'].max()
        # print(f'label is {label}')
        # if nan, then set to 0
        if np.isnan(label):
            label = 0
        label = int(label)

        # add the impact scalar if needed
        if self.impact_scalar:
            # get the min impact scalar in the window among the impact scalars that are not 0
            # if impact scalar is not set it will NaN. Min will ignore NaNs if there is a non-NaN value
            impact_scalar = utterances['impact_scalar'].min()
            if np.isnan(impact_scalar):
                impact_scalar = 0.0
            impact_scalar = float(impact_scalar)
            return {
                'input_ids': tokens,
                'label': label,
                'impact_scalar': impact_scalar
            }

        return {'input_ids': tokens, 'label': label}


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
    return {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': None,
        'id2label': id2label,
        'label2id': label2id
    }


def get_change_point_dataset(tokenizer, args):
    # load the dataset
    df = pd.read_csv(os.path.join(
        args.data_dir, 'change_point_social_orientation_train_val_test.csv'),
                     index_col=0)

    # split into train, val, and test sets
    train_df = df[df['split'] == 'train'].reset_index(drop=True).reset_index()
    val_df = df[df['split'] == 'val'].reset_index(drop=True).reset_index()
    test_df = df[df['split'] == 'test'].reset_index(drop=True).reset_index()

    # put an index on the DFs
    train_df = train_df.set_index(['index', 'file_id'])
    val_df = val_df.set_index(['index', 'file_id'])
    test_df = test_df.set_index(['index', 'file_id'])

    label2id = {'Change Point': 1, 'No Change Point': 0}
    id2label = {v: k for k, v in label2id.items()}

    # if debug_pct is set, use a subset of the training data
    # TODO: Note that this is a little naive because we're not shuffling and
    # arbitrarily grabbing the first N rows. At least it won't break the index
    if args.debug_pct < 1.0:
        train_df = train_df[:int(args.debug_pct * len(train_df))]

    # create the datasets
    train_dataset = ChangePointDataset(
        train_df,
        tokenizer,
        window_size=args.window_size,
        impact_scalar=args.impact_scalar,
        social_orientation=args.social_orientation)
    val_dataset = ChangePointDataset(
        val_df,
        tokenizer,
        window_size=args.window_size,
        impact_scalar=args.impact_scalar,
        social_orientation=args.social_orientation)
    test_dataset = ChangePointDataset(
        test_df,
        tokenizer,
        window_size=args.window_size,
        impact_scalar=args.impact_scalar,
        social_orientation=args.social_orientation)

    return {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'id2label': id2label,
        'label2id': label2id
    }


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
    return {
        'train_dataset': tokenized_imdb['train'],
        'val_dataset': tokenized_imdb['test'],
        'test_dataset': None,
        'id2label': id2label,
        'label2id': label2id
    }


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
            shuffle=True,
            seed=args.seed)
    train_collate_fn = collate_fn if train_collate_fn is None else train_collate_fn
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.num_dataloader_workers,
        pin_memory=True,
        collate_fn=train_collate_fn,
        sampler=train_sampler)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_dataloader_workers,
        pin_memory=True,
        collate_fn=collate_fn)
    if test_dataset is not None:
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=args.num_dataloader_workers,
            pin_memory=True,
            collate_fn=collate_fn)
        return train_dataloader, val_dataloader, test_dataloader
    return train_dataloader, val_dataloader, None

def get_eval_dataloader(batch_size, num_workers, dataset):
    # TODO: generalize this to take in the args as the model for the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
    collate = DataCollatorWithPadding(tokenizer=tokenizer)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate)

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
    elif args.dataset == 'circumplex':
        # get the Circumplex dataset
        data = get_circumpex_dataset(tokenizer, args.data_dir, args.debug_pct)
    elif args.dataset == 'change-point':
        data = get_change_point_dataset(tokenizer, args)

    args.id2label = data['id2label']
    args.label2id = data['label2id']
    train_collate = None
    if args.triplet_loss:
        train_collate = triplet_collate(tokenizer, len(args.id2label))
    collate = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        args,
        collate,
        data['train_dataset'],
        data['val_dataset'],
        test_dataset=data['test_dataset'],
        train_collate_fn=train_collate)
    return train_dataloader, val_dataloader, test_dataloader, args


def dist_log(args, message):
    if args.distributed and args.local_rank == 0:
        logging.info(message)
    elif not args.distributed:
        logging.info(message)
    
def get_best_model(predictions_dir, modality):
    """Identifies the model and filtering method that produces the best
    performance on the specified modality, which is one of {'text', 'audio', 'video', 'all'}.}"""
    model_dirs = os.listdir(predictions_dir)
    # load val_results.json if present
    val_results = {}
    for model_dir in model_dirs:
        val_results_filepath = os.path.join(predictions_dir, model_dir, 'val_results.json')
        if os.path.exists(val_results_filepath):
            with open(val_results_filepath, 'r') as f:
                val_results[model_dir] = json.load(f)
    df = pd.DataFrame.from_dict(val_results, orient='index')#.reset_index().rename(columns={'index': 'model'})

    df = df.stack().to_frame().rename(columns={0: 'val_results'}).reset_index().rename(columns={'level_0':'model', 'level_1': 'filtering'})
    df = pd.concat([df.drop(columns=['val_results']), pd.json_normalize(df['val_results'])], axis=1)
    # group by model and select filtering method with the highest text score
    best_df = df.groupby('model').apply(lambda x: x.loc[x[modality].idxmax()]).reset_index(drop=True)
    # go with change-point-medium-reweight for now
    best_row = best_df.iloc[best_df['text'].idxmax()]
    return best_row # best_row['model'], best_row['filtering'], best_row['text'], best_row['audio'], best_row['video']#, best_row['all']

if __name__ == '__main__':
    predictions_dir = '/mnt/swordfish-pool2/ccu/predictions'
    print(get_best_model(predictions_dir, modality='video'))