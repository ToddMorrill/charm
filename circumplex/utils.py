"""Utility file storing commonly used functions."""
import os
from collections import defaultdict
import json

import pandas as pd

def load_ldc_annotations(anno_dir):
    anno_files = [os.path.join(anno_dir, x) for x in os.listdir(anno_dir) if x not in ['.DS_Store']]
    anno_dfs = {}
    for f in anno_files:
        filename = os.path.split(f)[-1]
        anno_dfs[filename] = pd.read_csv(f, sep='\t')

    # load segment information, which is in os.path.join(../anno_dir, docs)
    head, tail = os.path.split(anno_dir)
    segment_filepath = os.path.join(head, 'docs/segments.tab')
    segment_df = pd.read_csv(segment_filepath, delimiter='\t')
    
    # load versions_per_file.tab, which is necessary to capture negative
    # examples for change point
    versions_filepath = os.path.join(head, 'docs/versions_per_file.tab')
    versions_df = pd.read_csv(versions_filepath, delimiter='\t')
    return anno_dfs, segment_df, versions_df

def load_transcribed_files(asr_dirs, return_data=False):
    # create lists of all filepaths and file_ids
    asr_files = []
    file_ids = []
    files_by_dir = defaultdict(list)
    dir_by_file = {}
    for dir_ in asr_dirs:
        for f in os.listdir(dir_):
            if f.endswith('.json'):
                filepath = os.path.join(dir_, f)
                asr_files.append(filepath)
                file_ids.append(f.split('_')[0])
                group = os.path.join(*filepath.split(os.sep)[-3:-1])
                files_by_dir[group].append(filepath)
                dir_by_file[os.path.split(filepath)[-1]] = group
    
    if not return_data:
        return asr_files, file_ids, files_by_dir, dir_by_file
    # load json files
    raw_data = {}
    data_dfs = {}
    for f in asr_files:
        filename = os.path.split(f)[-1]
        with open(f, 'r') as fh:
            raw_data[filename] = json.load(fh)
            file_id = filename.split('_')[0]
            if 'asr_turn_lvl' in raw_data[filename]:
                data_dfs[file_id] = pd.DataFrame(raw_data[filename]['asr_turn_lvl'])
            else:
                data_dfs[file_id] = pd.DataFrame(raw_data[filename]['asr_preprocessed_turn_lvl'])
    return asr_files, file_ids, files_by_dir, dir_by_file, raw_data, data_dfs

def load_translated_files(translation_dir, return_data=False):
    # load transcriptions/translations
    data = {}
    data_dfs = {}
    translation_files = []
    for x in os.listdir(translation_dir):
        if x.endswith('.json'):
            filepath = os.path.join(translation_dir, x)
            translation_files.append(filepath)
            file_id = x.split('.')[0]
            if return_data:
                with open(filepath, 'r', encoding='utf-8') as fp:
                    data[file_id] = json.load(fp)
                data_dfs[file_id] = pd.DataFrame(data[file_id]['asr_turn_lvl'])
    if return_data:
        return translation_files, data_dfs, data
    return translation_files

