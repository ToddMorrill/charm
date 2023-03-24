"""This module prepares predictions and annotations the specified video.

Examples:
    $ python video.py \
        --data-dir ~/Documents/data/charm \
        --file-id M010047AU
"""
import argparse
import os
import shutil
import simplejson

import pandas as pd

from charm.data import utils

# norms map
NORMS_MAP = {
    101: 'doing apology',
    102: 'doing criticism',
    103: 'doing greeting',
    104: 'doing request',
    105: 'doing persuasion',
    106: 'doing thanks',
    107: 'doing taking leave'
}


def valence_arousal(anno_dfs, segment_df, video):
    valence_df = anno_dfs['valence_arousal.tab']

    # filter out noann rows
    valence_df = valence_df[
        valence_df['valence_continuous'] != 'noann'].reset_index(drop=True)

    # convert cols to floats
    valence_convert_cols = [
        'valence_continuous', 'valence_binned', 'arousal_continuous',
        'arousal_binned'
    ]
    valence_df.loc[:, valence_convert_cols] = valence_df[
        valence_convert_cols].astype(float)

    # drop the user_id column
    valence_df.drop(columns=['user_id'], inplace=True)

    # average valence over annotators
    valence_df = valence_df.groupby(['file_id', 'segment_id'],
                                    as_index=False).mean()

    valence_df = valence_df.merge(segment_df.drop(columns=['file_id']),
                                  how='left',
                                  on='segment_id')

    valence_df = valence_df[valence_df['file_id'] == video['file_uid']]

    return valence_df


def emotions(anno_dfs, segment_df, video):
    emotions_df = anno_dfs['emotions.tab']

    # drop emotion == 'none'
    emotions_df = emotions_df[(emotions_df['emotion'] != 'none')
                              & (emotions_df['emotion'] != 'noann')]

    # split emotion string and explode df
    emotions_df['emotion'] = emotions_df['emotion'].apply(
        lambda x: x.split(','))
    emotions_df = emotions_df.explode(column='emotion')

    # drop user_id
    emotions_df.drop(columns=['user_id', 'multi_speaker'], inplace=True)

    # group by file_id, segment_id, create a set of emotions
    emotions_df = emotions_df.groupby(['file_id', 'segment_id'],
                                      as_index=False).agg(set)

    # convert set to ordered list
    emotions_df['emotion'] = emotions_df['emotion'].apply(
        lambda x: ', '.join(sorted(list(x))))

    emotions_df = emotions_df.merge(segment_df.drop(columns=['file_id']),
                                    how='left',
                                    on='segment_id')

    emotions_df = emotions_df[emotions_df['file_id'] == video['file_uid']]

    return emotions_df


def norms(anno_dfs, segment_df, video):
    norms_df = anno_dfs['norms.tab']

    norms_df = norms_df[(norms_df['norm'] != 'none')
                        & (norms_df['norm'] != 'noann')]

    norms_df.drop(columns='user_id', inplace=True)

    norms_df = norms_df.groupby(['file_id', 'segment_id'],
                                as_index=False).agg(list)

    norms_df = norms_df.merge(segment_df.drop(columns=['file_id']),
                              how='left',
                              on='segment_id')

    norms_df = norms_df[norms_df['file_id'] == video['file_uid']]

    # pull in norms map
    norms_df['norm'] = norms_df['norm'].apply(
        lambda x: [NORMS_MAP[int(y)] for y in x])

    return norms_df


def changepoint(anno_dfs, segment_df, video):
    changepoint_df = anno_dfs['changepoint.tab']

    changepoint_df.drop(columns='user_id', inplace=True)

    changepoint_df = changepoint_df[changepoint_df['file_id'] ==
                                    video['file_uid']]

    return changepoint_df


def save_annotations(valence_df, emotions_df, norms_df, changepoint_df,
                     out_filepath):
    video_annos = {}
    video_annos['valence_arousal'] = valence_df.to_dict('records')
    video_annos['emotion'] = emotions_df.to_dict('records')
    video_annos['norms'] = norms_df.to_dict('records')
    video_annos['changepoint'] = changepoint_df.to_dict('records')

    with open(out_filepath, 'w') as f:
        simplejson.dump(video_annos, f)


def save_predictions(data_dir, file_id, out_dir):
    cd = 'CCU_P1_TA1_CD_COL_LDC2022E22-V1_20221128_150559'
    ad = 'CCU_P1_TA1_AD_COL_LDC2022E22-V1_20221123_141038'
    vd = 'CCU_P1_TA1_VD_COL_LDC2022E22-V1_20221123_144304'
    ed = 'CCU_P1_TA1_ED_COL_LDC2022E22-V1_20221128_150401'
    nd = 'CCU_P1_TA1_ND_COL_LDC2022E22-V1_20221129_101234'
    # load predictions for chosen file
    submissions_dir = os.path.join(data_dir,
                                   'transformed/predictions/submissions')
    prediction_dfs = {}
    for pred_dir, pred_type in [(cd, 'cd'), (ad, 'ad'), (vd, 'vd'), (ed, 'ed'),
                                (nd, 'nd')]:
        pred_file = os.path.join(submissions_dir, pred_dir, f"{file_id}.tab")
        prediction_dfs[pred_type] = pd.read_csv(pred_file, delimiter='\t')

    # merge valence and arousal
    # NB: this won't always work if timestamps aren't aligned
    vd_ad_preds_df = prediction_dfs['ad'].merge(prediction_dfs['vd'],
                                                how='inner',
                                                on=['file_id', 'start', 'end'])

    # group ['file_id', 'start', 'end'] and aggregrate to a list
    nd_preds_df = prediction_dfs['nd'].groupby(['file_id', 'start', 'end'],
                                               as_index=False).agg(list)

    temp_df = prediction_dfs['ed'].groupby(['file_id', 'start', 'end'],
                                           as_index=False).agg(list)

    # check if any intervals have multiple emotions
    (temp_df['emotion'].apply(lambda x: len(x)) > 1).sum()

    # pull in norms map
    nd_preds_df['norm'] = nd_preds_df['norm'].apply(
        lambda x: [NORMS_MAP[int(y)] for y in x])

    video_preds = {}
    video_preds['valence_arousal'] = vd_ad_preds_df.to_dict('records')
    video_preds['emotion'] = prediction_dfs['ed'].to_dict('records')
    video_preds['norms'] = nd_preds_df.to_dict('records')
    video_preds['changepoint'] = prediction_dfs['cd'].to_dict('records')

    out_filepath = os.path.join(out_dir, f'{file_id}_predictions.json')
    with open(out_filepath, 'w') as f:
        simplejson.dump(video_preds, f)


def main(args):
    meta_filepath = os.path.join(args.data_dir, 'transformed/metadata.csv')
    meta_df = pd.read_csv(meta_filepath)

    # make directory for all json files
    out_dir = args.file_id
    os.makedirs(out_dir, exist_ok=True)
    video_df = meta_df[meta_df['file_uid'] == args.file_id]
    if len(meta_df) == 0:
        raise ValueError('No metadata found for file ID: {}'.format(
            args.file_id))
    metadata_filepath = os.path.join(out_dir, f'{args.file_id}_metadata.json')
    with open(metadata_filepath, 'w') as f:
        simplejson.dump(video_df.iloc[0].to_dict(), f, ignore_nan=True)

    # load annotations for this file
    mini_eval_filepath = os.path.join(
        args.data_dir,
        'raw/LDC2023E01_CCU_TA1_Mandarin_Chinese_Mini_Evaluation_Annotation_Unsequestered/'
    )
    result = utils.load_ldc_annotation(mini_eval_filepath)
    anno_dfs, segment_df, version_df = result
    valence_df = valence_arousal(anno_dfs, segment_df, video_df.iloc[0])
    emotions_df = emotions(anno_dfs, segment_df, video_df.iloc[0])
    norms_df = norms(anno_dfs, segment_df, video_df.iloc[0])
    changepoint_df = changepoint(anno_dfs, segment_df, video_df.iloc[0])

    # save annotations
    out_filepath = os.path.join(out_dir, f'{args.file_id}_annotations.json')
    save_annotations(valence_df, emotions_df, norms_df, changepoint_df,
                     out_filepath)
    
    # save predictions
    save_predictions(args.data_dir, args.file_id, out_dir)

    
    # translate
    video_filepath = os.path.join(args.data_dir, 'raw/LDC2022E22_CCU_TA1_Mandarin_Chinese_Mini_Evaluation_Source_Data/data/video', f'{args.file_id}.mp4.ldcc')
    transcription_output = os.path.join(out_dir, f'{args.file_id}.mp4.json')
    transcript = utils.transcribe(video_filepath, transcription_output, task='translate')

    # zip up directory
    shutil.make_archive(out_dir, 'zip', out_dir)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-dir',
        help='The data directory where metadata.csv is stored.',
        type=str,
        required=True)
    parser.add_argument('--file-id',
                        help='The file ID to be processed.',
                        type=str,
                        required=True)
    args = parser.parse_args()
    main(args)