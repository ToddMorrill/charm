"""Utility file storing commonly used functions."""
import os
from collections import defaultdict
import logging
import json

import pandas as pd

# add in easy to understand data types
MODALITIES = {
    '.mp4.ldcc': 'video',
    '.ltf.xml': 'text',
    '.psm.xml': 'text',
    '.flac.ldcc': 'audio'
}


def load_release_metadata(raw_data_dir):
    """Loads metadata related to all available releases stored in
    raw_data_dir."""
    # release folders
    releases = {
        'R1':
        'LDC2022E11_CCU_TA1_Mandarin_Chinese_Development_Source_Data_R1',
        'R2':
        'LDC2022E19_CCU_TA1_Mandarin_Chinese_Development_Source_Data_R2_V2.0',
        'R3':
        'LDC2022E20_CCU_TA1_Mandarin_Chinese_Development_Source_Data_R3_V1.0',
        'Mini-Eval':
        'LDC2022E22_CCU_TA1_Mandarin_Chinese_Mini_Evaluation_Source_Data'
    }
    release_filepaths = {}
    for release in releases:
        release_filepaths[release] = os.path.join(raw_data_dir,
                                                  releases[release],
                                                  'docs/file_info.tab')

    release_dfs = {}
    for release in release_filepaths:
        if not os.path.exists(release_filepaths[release]):
            logging.warning(
                f'Release {release_filepaths[release]} does not exist and is being skipped.'
            )
            continue
        temp_df = pd.read_csv(release_filepaths[release], delimiter='\t')
        temp_df.insert(0, column='release', value=release)
        release_dfs[release] = temp_df

    return release_dfs


def load_ldc_annotations(raw_data_dir):
    logging.info('Loading LDC annotations')
    # anno folders
    annos = {
        'Annotation-1':
        'LDC2022E18_CCU_TA1_Mandarin_Chinese_Development_Annotation_V3.0',
        'Mini-Eval-Annotations':
        'LDC2023E01_CCU_TA1_Mandarin_Chinese_Mini_Evaluation_Annotation_Unsequestered'
    }
    anno_filepaths = {}
    for anno in annos:
        anno_filepaths[anno] = os.path.join(raw_data_dir, annos[anno], 'data')

    anno_dfs = {}
    for anno in anno_filepaths:
        if not os.path.exists(anno_filepaths[anno]):
            logging.warning(
                f'Release {anno_filepaths[anno]} does not exist and is being skipped.'
            )
            continue
        anno_dfs_, segment_df, versions_df = load_ldc_annotation(
            anno_filepaths[anno])
        anno_dfs[anno] = {
            'anno_dfs': anno_dfs_,
            'segment_df': segment_df,
            'versions_df': versions_df
        }

    return anno_dfs


def load_ldc_annotation(anno_dir):
    """Loads LDC annotation data stored in anno_dir."""
    # if anno_dir doesn't end with 'data', add it
    if not anno_dir.endswith('data'):
        anno_dir = os.path.join(anno_dir, 'data')
    anno_files = [
        os.path.join(anno_dir, x) for x in os.listdir(anno_dir)
        if x not in ['.DS_Store']
    ]
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


def load_transcribed_files(transcriptions_dir, return_data=False):
    """Loads automatic speech recognition (ASR) transcripts stored in
    asr_dirs."""
    logging.info('Loading transcribed files')
    asr_dirs = [
        'LDC Mini-Eval Release', 'LDC Release 2', 'LDC Release 3',
        'LDC Release 1'
    ]
    asr_dirs = [
        os.path.join(transcriptions_dir, asr_dir) for asr_dir in asr_dirs
    ]

    # get all subdirs for each release folder except release 1
    asr_subdirs = []
    for asr_dir in asr_dirs[:-1]:
        for item in os.listdir(asr_dir):
            subpath = os.path.join(asr_dir, item)
            if os.path.isdir(subpath):
                asr_subdirs.append(subpath)

    # manually add in Release 1 subdirs
    asr_subdirs.append(os.path.join(asr_dirs[-1], 'audio_processed'))
    asr_subdirs.append(os.path.join(asr_dirs[-1], 'video_processed'))

    # create lists of all filepaths and file_ids
    asr_files = []
    file_ids = []
    files_by_dir = defaultdict(list)
    dir_by_file = {}
    for dir_ in asr_subdirs:
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
                data_dfs[file_id] = pd.DataFrame(
                    raw_data[filename]['asr_turn_lvl'])
            else:
                data_dfs[file_id] = pd.DataFrame(
                    raw_data[filename]['asr_preprocessed_turn_lvl'])
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
