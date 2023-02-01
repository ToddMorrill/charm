"""This module is responsible for creating the metadata index into all LDC data.

Examples:
    $ python -m charm.data.metadata \
        --raw-data-dir ~/Documents/data/charm/raw
"""
import argparse
import logging
import os

import pandas as pd
import numpy as np

from . import utils


def merge_versions(anno_dfs, meta_df):
    # create version info (i.e. number of unique annotation versions per file to capture that there were multiple annotators)
    versions_df = pd.concat({k: anno_dfs[k]['versions_df']
                             for k in anno_dfs},
                            ignore_index=True)
    # make sure no overlap between releases
    assert versions_df['file_id'].nunique() == len(versions_df)
    # merge version info in
    meta_df = pd.merge(meta_df,
                       versions_df.rename(columns={'file_id': 'file_uid'}),
                       left_on='file_uid',
                       right_on='file_uid',
                       how='left')
    return meta_df


def merge_segments(anno_dfs, meta_df):
    # create segment info (i.e. segments annotated)
    segment_df = pd.concat({k: anno_dfs[k]['segment_df']
                            for k in anno_dfs},
                           ignore_index=True)

    seg_start_df = segment_df.groupby(
        'file_id')['start'].min().to_frame().reset_index()
    seg_end_df = segment_df.groupby(
        'file_id')['end'].max().to_frame().reset_index()

    seg_df = pd.merge(seg_start_df, seg_end_df, on='file_id')

    # some of these files have very long contigous stretches of annotations
    # spot checking reveals that they are contiguous but some may not be
    # also NB: the segments are not perfectly contiguous (there are typically a few gap seconds for music, etc.)
    seg_df[(seg_df['end'] - seg_df['start']) > 310]

    # merge segments in
    meta_df = pd.merge(meta_df,
                       seg_df.rename(columns={'file_id': 'file_uid'}),
                       left_on='file_uid',
                       right_on='file_uid',
                       how='left')
    return meta_df


def merge_transcription_info(asr_data, meta_df):
    asr_files, file_ids, files_by_dir, dir_by_file, raw_asr_data, asr_data_dfs = asr_data

    asr_len_data = []
    for key in asr_data_dfs:
        asr_len_data.append((key, len(asr_data_dfs[key])))

    # add transcription status
    asrd_df = pd.DataFrame(asr_len_data,
                           columns=['file_uid', 'utterance_count'])
    asrd_df['transcribed'] = True

    meta_df = pd.merge(meta_df,
                       asrd_df.rename(columns={'file_id': 'file_uid'}),
                       left_on='file_uid',
                       right_on='file_uid',
                       how='left')
    return meta_df


def data_checks(meta_df):
    # sanity check the numbers found in the README.txt files
    r1_count = 1143 + (
        488 * 2
    )  # text files have 2 corresponding files, need to double count the text files
    r2_count = 4914 + 0 + 0 # video + audio + text
    r3_count = 6040 + 644 + 0  # video + audio + text
    mini_eval_count = 1501 + 212 + (384 * 2)  # video + audio + text*2
    file_count = r1_count + r2_count + r3_count + mini_eval_count
    assert file_count == len(meta_df)

    # make sure no overlap between releases
    file_by_release_df = meta_df.groupby([
        'file_uid', 'release'
    ]).agg(count=('catalog_id', 'count')).unstack().fillna(value=0)
    assert (file_by_release_df.astype(bool).sum(axis=1) > 1).sum() == 0


def save_metadata(meta_df, transformed_dir):
    # save to transformed dir
    meta_filepath = os.path.join(transformed_dir, 'metadata.csv')
    logging.info(f'Saving {meta_filepath}')
    meta_df.to_csv(meta_filepath, index=False)


def main(args):
    logging.info('Loading release metadata.')
    release_dfs = utils.load_release_metadata(args.raw_data_dir)
    meta_df = pd.concat(release_dfs, ignore_index=True)
    data_checks(meta_df)
    # add in easy to understand data types
    modalities = {
        '.mp4.ldcc': 'video',
        '.ltf.xml': 'text',
        '.psm.xml': 'text',
        '.flac.ldcc': 'text'
    }
    meta_df['modality'] = meta_df['data_type'].apply(lambda x: modalities[x])

    # load annotation data
    anno_dfs = utils.load_ldc_annotations(args.raw_data_dir)
    meta_df = merge_versions(anno_dfs, meta_df)
    meta_df = merge_segments(anno_dfs, meta_df)

    # add transcription status
    head, tail = os.path.split(args.raw_data_dir)
    # https://drive.google.com/drive/u/0/folders/1rhRJhBgtBuMSpcWn8nQHWmlMUGqfmAba
    transcriptions_dir = os.path.join(head, 'transformed/transcriptions')
    asr_data = utils.load_transcribed_files(transcriptions_dir,
                                            return_data=True)
    meta_df = merge_transcription_info(asr_data, meta_df)

    transformed_dir = os.path.join(head, 'transformed')

    col_order = [
        'release', 'catalog_id', 'file_uid', 'url', 'modality', 'start', 'end',
        'transcribed', 'utterance_count', 'emotion_count',
        'valence_arousal_count', 'norms_count', 'changepoint_count',
        'emotions_count', 'data_type', 'lang_id_manual', 'wrapped_md5',
        'unwrapped_md5', 'download_date', 'content_date', 'status_in_corpus',
        'legacy_catalog_id', 'original_file_id', 'type', 'file_path', 'length',
        'version'
    ]
    meta_df = meta_df[col_order]
    data_checks(meta_df)
    save_metadata(meta_df, transformed_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--raw-data-dir',
        required=True,
        help='The directory where all LDC release folders are stored.')
    args = parser.parse_args()
    main(args)