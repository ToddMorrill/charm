"""This module creates a folder ./index_files/ and the following index files:
A) ./index_files/COMPLETE.system_input.index.tab that lists all the files
present in the ./data folder. This index file has 4 columns:
    1. file_id
    2. type: one of {audio, video, text}
    3. file_path: e.g. ./data/text/1.txt, defaults to .data
    4. length: this is not used by the scoring system and defaults to 1000

B) ./index_files/COMPLETE.scoring.index.tab that lists all the file_ids that
should be scored. This index file has 1 columns:
    1. file_id

Examples:
    $ python -m charm.eval.create_system_index \
        --data-dir ~/Documents/data/charm/raw/LDC2023E01_CCU_TA1_Mandarin_Chinese_Mini_Evaluation_Annotation_Unsequestered
"""
import argparse
import os

import pandas as pd
from ..data import utils


def main(args):
    # create the index_files folder and system input index file
    data_dir = args.data_dir
    index_file = os.path.join(data_dir, 'index_files',
                              'COMPLETE.system_input.index.tab')
    os.makedirs(os.path.dirname(index_file), exist_ok=True)

    # load file_info.tab to get data type, fail if it doesn't exist
    if not os.path.exists(os.path.join(data_dir, 'docs/file_info.tab')):
        raise FileNotFoundError(
            f'file_info.tab not found in {os.path.join(data_dir, "docs")}')
    file_info_df = pd.read_csv(os.path.join(data_dir, 'docs/file_info.tab'),
                               sep='\t')

    # move a copy of file_info.tab to file_info.tab.bak to have a backup
    # if it doesn't already exist
    if not os.path.exists(os.path.join(data_dir, 'docs/file_info.tab.bak')):
        file_info_df.to_csv(os.path.join(data_dir, 'docs/file_info.tab.bak'),
                            sep='\t',
                            index=False)

    # update file_info.tab with necessary columns
    file_info_df['type'] = file_info_df['data_type'].apply(
        lambda x: utils.MODALITIES[x])
    file_info_df['file_path'] = './data'
    file_info_df['length'] = 1000
    file_info_df.to_csv(os.path.join(data_dir, 'docs/file_info.tab'),
                        index=False,
                        sep='\t')

    # rename file_uid to file_id
    file_info_df.rename(columns={'file_uid': 'file_id'}, inplace=True)
    # generate necessary columns
    file_info_df['type'] = file_info_df['data_type'].apply(
        lambda x: utils.MODALITIES[x])
    file_info_df['file_path'] = './data'
    file_info_df['length'] = 1000
    file_info_df.drop_duplicates(subset=['file_id'], inplace=True)
    file_info_df[['file_id', 'type', 'file_path',
                  'length']].to_csv(index_file, sep='\t', index=False)

    # create the scoring index file for changepoint
    scoring_index_file = os.path.join(data_dir, 'index_files',
                                      'COMPLETE.scoring.index.tab')
    # only keep file_ids labeled for changepoint
    anno_dfs, segment_df, versions_df = utils.load_ldc_annotation(data_dir)
    annotated_file_ids = versions_df[versions_df['changepoint_count'] > 0]['file_id'].unique()
    anno_subset_df = file_info_df[file_info_df['file_id'].isin(annotated_file_ids)]
    anno_subset_df[['file_id']].to_csv(scoring_index_file, sep='\t', index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',
                        help='Directory where the labeled data is stored.',
                        type=str,
                        required=True)
    args = parser.parse_args()
    main(args)