"""This module prepares text conversations to be sent to GPT for social
orientation annotation. Processing is done on a per document basis so
that everything can be parallelized.

Examples:
    $ python prepare_text_conversations.py \
        --sample
"""
import argparse
import os
import logging
from tqdm import tqdm
import warnings
import shutil

import pandas as pd
import tiktoken

import utils

logging.basicConfig(level=logging.INFO)

# filter pandas warning
warnings.simplefilter('ignore', DeprecationWarning)

GPT_3_5_TURBO_TOKEN_LIMIT = 4096
GPT_4_TURBO_TOKEN_LIMIT = 8192

def load_data(args):
    head, tail = os.path.split(args.input_pickle)
    cache_filepath = os.path.join(head, tail.split('.')[0] + '.csv')
    if args.use_cache and os.path.exists(cache_filepath) and (not args.sample):
        logging.info('Using cached CSV file.')
        df = pd.read_csv(cache_filepath)
        return df
    
    # otherwise, load the pickle file and convert to df
    logging.info('Loading pickle file..')
    data = utils.load_pickle(args.input_pickle)
    # convert to Pandas for easier processing
    dfs = []
    for file_id in tqdm(data.keys()):
        if not data[file_id]['processed']:
            continue
        
        # if sampling, only use 3 file ids
        if args.sample and file_id not in ['M01004WNG', 'M010008RY', 'M01000HIQ']:
            continue
        # TODO: may need to pull in additional metadata
        # if data_type == 'text', don't need whisper key
        if data[file_id]['data_type'] == 'text':
            temp_df = pd.DataFrame(data[file_id]['utterances'])
        else:
            temp_df = pd.DataFrame(data[file_id]['utterances'][args.transcript])
        # include additional metadata
        temp_df['file_id'] = file_id
        temp_df['splits'] = [data[file_id]['splits'] for _ in range(len(temp_df))]
        temp_df['data_type'] = data[file_id]['data_type']
        dfs.append(temp_df)
    df = pd.concat(dfs)
    # drop the audio/video columns
    df = df.drop(columns=['audio_files', 'video_frames'])
    if not args.sample:
        # save to CSV for faster loading if we use the cache
        df.to_csv(cache_filepath, index=False)
    return df

def id_speakers_convo(group_df):
    """Give each speaker a numerical identifier."""
    if 'participant' not in group_df.columns:
        group_df['participant_id'] = 'unknown'
        return group_df
    
    # fillna with unknown
    group_df['participant'] = group_df['participant'].fillna('unknown')

    speaker_map = {'unknown': 'unknown'}
    participants = set(group_df['participant'].unique())
    # exclude unknown so we don't give it an ID number
    if 'unknown' in participants:
        participants.remove('unknown')
    for idx, participant in enumerate(participants):
        speaker_map[participant] = idx + 1

    # apply speaker map to the participant column
    group_df['participant_id'] = group_df['participant'].apply(lambda x: speaker_map[x])
    return group_df

def add_utterance_id(group_df):
    """Add a unique identifier for each utterance."""
    group_df['utterance_id'] = list(range(len(group_df)))
    # add 1 to the utterance id so that it starts at 1
    group_df['utterance_id'] = group_df['utterance_id'] + 1
    return group_df

def create_line(row, truncate=100):
    """Creates conversation row in markdown format. Any utterance over 100 characters is probably nonsense."""
    content = f"| {row['utterance_id']} | {row['participant_id']} | {row['text'][:truncate]} |"
    return content

def pandas_to_markdown(df):
    """Converts each row of a pandas dataframe to markdown delimited by | with a single space on each side.
    There is also a row of dashes between the header and the table body.
    """
    header = '| ' + ' | '.join(df.columns) + ' |'
    dashes = '| ' + ' | '.join(['---' for _ in df.columns]) + ' |'
    data = '\n'.join(['| ' + ' | '.join([str(x) for x in row]) + ' |' for row in df.values])
    return '\n'.join([header, dashes, data])

def assign_chunks(df, max_input_length, overlap=10):
    """Assigns each row to a contiguous chunk of text that is less than the max_input_length. Just before a chunk will exceed the max_input_length, a new chunk is created.
    """
    start = 0
    end = max_input_length
    idx_end = 0
    chunk_id = 0
    dfs = []
    while idx_end != df.iloc[-1].name:
        # retrieve slice that's between the current start and current end
        chunk_df = df[(df['encoding_length_cumsum'] > start) & (df['encoding_length_cumsum'] <= end)]
        idx_start = chunk_df.iloc[0].name
        idx_end = chunk_df.iloc[-1].name
        
        # assign chunk_id to the slice
        # df.loc[idx_start:idx_end, 'chunk_id'] = chunk_id
        chunk_df.loc[idx_start:idx_end, 'chunk_id'] = chunk_id
        dfs.append(chunk_df)
        chunk_id += 1

        # NB: overlap MUST be set so that the maximum number of tokens in an
        # overlap length sequence is less than the max_input_length
        # e.g. if texts are truncated to 100 and max_input_length is 1000, then
        # overlap must be 10 or less
        if idx_end != df.iloc[-1].name:
            idx_end -= overlap
        
        # update start and end
        start = chunk_df.loc[idx_end]['encoding_length_cumsum']
        end = start + max_input_length

    # return df
    return pd.concat(dfs)


def group_cumsum(group_df):
    group_df['encoding_length_cumsum'] = group_df['encoding_length'].cumsum()
    return group_df

def pipeline_part_1(args):
    head, tail = os.path.split(args.input_pickle)
    cache_filepath = os.path.join(head, tail.split('.')[0] + '.csv')
    if args.use_cache and os.path.exists(cache_filepath) and (not args.sample):
        df = pd.read_csv(cache_filepath)
        # check if encoding_length column is present
        if 'encoding_length' in df.columns:
            logging.info('Using cached CSV file.')
            return df
    
    # otherwise, load the pickle file and convert to df
    df = load_data(args)
    
    # convert participant IDs to speaker numbers
    logging.info('Converting participant IDs to speaker numbers..')
    df = df.groupby('file_id', group_keys=False).apply(id_speakers_convo)
    
    # add utterance_id
    # sort by start to be safe
    df = df.sort_values(by=['file_id', 'start'], ascending=True)
    df = df.reset_index(drop=True)
    # create utterance ids per file_id
    df = df.groupby('file_id', group_keys=False).apply(add_utterance_id)

    # truncate long utterances generously above the 75th percentile of 11 characters
    truncate_length = 100
    # create GPT line
    logging.info('Creating GPT lines..')
    df['gpt_line'] = df.apply(create_line, axis=1, truncate=truncate_length)

    # record encoding length
    logging.info('Recording encoding length..')
    encoding = tiktoken.encoding_for_model(args.model)
    df['encoding_length'] = df['gpt_line'].apply(lambda x: len(encoding.encode(x)))

    if not args.sample:
        # save to CSV for faster loading if we use the cache
        df.to_csv(cache_filepath, index=False)
    return df

def assign_splits(split_set):
    if 'INTERNAL_TRAIN' in split_set:
        return 'INTERNAL_TRAIN'
    elif 'INTERNAL_VAL' in split_set:
        return 'INTERNAL_VAL'
    elif 'INTERNAL_TEST' in split_set:
        return 'INTERNAL_TEST'
    elif 'EVALUATION_LDC2023E07' in split_set:
        return 'EVALUATION_LDC2023E07'

def create_gpt_data(group_df, system_prompt, prompt):
    # combine all gpt lines into a single string separated by newlines
    gpt_lines = '\n'.join(group_df['gpt_line'].values)
    final_prompt = prompt + gpt_lines
    # format GPT messages
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': final_prompt},
        ]
    return messages

def main(args):
    # load data
    df = pipeline_part_1(args)

    # load prompt
    with open(args.prompt_filepath, 'r') as f:
        prompt = f.read()
    
    encoding = tiktoken.encoding_for_model(args.model)
    prompt_length = len(encoding.encode(prompt))
    logging.info(f'Prompt length: {prompt_length}')
    token_limit = GPT_3_5_TURBO_TOKEN_LIMIT if '3.5' in args.model else GPT_4_TURBO_TOKEN_LIMIT
    # subtract prompt length from token limit and use specified portion for input tokens
    max_input_length = int((token_limit - prompt_length) * (args.data_token_pct))
    logging.info(f'Max conversation token length (excluding prompt): {max_input_length}')
    logging.info(f'Max generative length: {token_limit - max_input_length - prompt_length}')

    # create cumulative sum of encoding lengths to easier chunking
    df = df.groupby('file_id', group_keys=False).apply(group_cumsum)
    
    # assign utterances to disjoint chunks
    # TODO: can generalize this to have overlapping chunks
    # but not going to worry about micro-optimizations for now
    logging.info('Assigning utterances to chunks..')
    df = df.groupby('file_id', group_keys=False).apply(assign_chunks, max_input_length=max_input_length, overlap=args.overlap)

    # split by train/val/test/eval
    data = utils.load_pickle(args.input_pickle)
    
    # TODO: this should be dynamic based on content in the data pickle
    splits = {'EVALUATION_LDC2023E07', 'INTERNAL_TEST', 'INTERNAL_TRAIN', 'INTERNAL_VAL'}

    # assign single split in this order of priority: 'INTERNAL_TRAIN', 'INTERNAL_VAL', 'INTERNAL_TEST', 'EVALUATION_LDC2023E07'
    df['final_split'] = df['splits'].apply(assign_splits)
    
    # create GPT data
    logging.info('Creating GPT data..')
    system_prompt = 'You are a helpful assistant.'
    gpt_df = df.groupby(['file_id', 'chunk_id', 'final_split']).apply(create_gpt_data, system_prompt=system_prompt, prompt=prompt)
    gpt_df = gpt_df.reset_index().rename(columns={0: 'messages'})
    
    # shuffle so that file_ids are 

    # assign a hash bucket ID to each file_id within each split
    # so that we have chunks of about 100 files per bucket
    # NB: pandas orders the file_ids alphabetically, so assuming that file_ids are
    # randomly given, this should randomly assign file_ids to buckets
    gpt_df['hash_bucket_id'] = gpt_df.groupby(['final_split', 'file_id']).ngroup() // 100

    # clear existing contents of output directory
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # save all contents for each final_split, hash_bucket_id combo
    # to a JSONL file
    logging.info('Saving GPT data..')
    # if sample mode, just save the file directly to the output directory
    if args.sample:
        gpt_df.drop(columns=['final_split', 'hash_bucket_id']).to_json(os.path.join(args.output_dir, 'sample.jsonl'), orient='records', lines=True)
    else:
        for split in splits:
            num_buckets = gpt_df[(gpt_df['final_split'] == split)]['hash_bucket_id'].nunique()
            for bucket_id in range(num_buckets):
                split_df = gpt_df[(gpt_df['final_split'] == split) & (gpt_df['hash_bucket_id'] == bucket_id)]
                if len(split_df) > 0:
                    # drop final_split and hash_bucket_id columns
                    split_df = split_df.drop(columns=['final_split', 'hash_bucket_id'])
                    if split == 'INTERNAL_TRAIN':
                        breakpoint()
                    # save to JSONL in output directory for each split
                    output_dir = os.path.join(args.output_dir, split)
                    os.makedirs(output_dir, exist_ok=True)
                    split_df.to_json(os.path.join(output_dir, f'{bucket_id}.jsonl'), orient='records', lines=True)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-pickle', type=os.path.expanduser, default='~/Documents/data/charm/transformed/tm3229-cache-slim.pkl',
                        help='Path to the pickle file containing the change point data.')
    parser.add_argument('--output-pickle', type=os.path.expanduser, default='~/Documents/data/charm/transformed/tm3229-cache-slim-gpt.pkl',
                        help='Path to the output pickle file.')
    parser.add_argument('--output-dir', type=str, default='./data', help='Path to the output directory where JSONL files will be stored.')
    parser.add_argument('--prompt-filepath', type=str, default='./prompt.txt', help='Path to the GPT instruction prompt file.')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo', help='Which GPT model to use.')
    parser.add_argument('--transcript', type=str, default='whisper', help='Which transcript to use (e.g. whisper, azure, wav2vec).')
    parser.add_argument('--use-cache', action='store_true', help='Whether to use the cached CSV file.')
    parser.add_argument('--data-token-pct', type=float, default=0.5, help='What percentage of the token limit to (after removing the prompt count) to use for data. The remainder (i.e. 1 - data-token-pct) will be available to the GPT model for generation. This may be need to set through a bit of trial and error by 0.5 is a good starting point.')
    parser.add_argument('--sample', action='store_true', help='Whether to sample a couple file_ids for the purposes of quickly developing this module.')
    parser.add_argument('--overlap', type=int, default=10, help='Number of overlapping utterances to include in GPT calls.')
    args = parser.parse_args()
    main(args)