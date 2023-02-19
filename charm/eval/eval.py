"""This script implements evaluation metrics for the change point detection task.

Examples:
    $ python -m charm.eval.eval \
        --input-dir ~/Documents/data/charm/transformed/predictions/CCU_P1_TA1_CD_COL_LDC2022E22-V1_20221128_150559 \
        --reference-dir ~/Documents/data/charm/raw/LDC2023E01_CCU_TA1_Mandarin_Chinese_Mini_Evaluation_Annotation_Unsequestered \
        --output-dir ~/Documents/data/charm/transformed/eval
"""
import argparse
from collections import namedtuple
import multiprocessing
import os
import logging

import pandas as pd
import numpy as np
from sklearn.metrics import auc
from tqdm import tqdm

from ..data import utils


def load_data(input_dir, reference_dir):
    """Loads the data from the input and reference directories."""
    # load system_output.index.tab file to determine which files are valid
    system_output_df = pd.read_csv(os.path.join(input_dir,
                                                'system_output.index.tab'),
                                   sep='\t')
    system_output_df = system_output_df[system_output_df['is_processed']]
    # load system predictions
    system_predictions = {}
    for file_uid in system_output_df['file_id']:
        system_predictions[file_uid] = pd.read_csv(os.path.join(
            input_dir, f'{file_uid}.tab'),
                                                   sep='\t')

    # load reference data
    anno_dfs, segment_df, versions_df = utils.load_ldc_annotation(
        reference_dir)
    return system_predictions, anno_dfs, segment_df, versions_df


def mapping(system_predictions: list[dict], reference_data: list[dict],
            delta: float) -> list[(dict, dict)]:
    """Maps the predictions to the reference data.
    For each change point:
        - Build a list of potentially mappable system/reference instance pairs where
        change points are mappable if the system prediction and reference are within delta time of each other.
        - Sort pairs by decreasing detection score.
        - While the pair list is not empty:
            - The top pair (system_x, reference_y) is added to the correct detection pair list
            - Remove unused pairs for system_x
            - Remove unused pairs for reference_y
    """
    # double for-loop to build the list of potentially mappable pairs
    pairs = []
    for system_x in system_predictions:
        for reference_y in reference_data:
            if abs(system_x['timestamp'] - reference_y['timestamp']) <= delta:
                pairs.append((system_x, reference_y))

    # sort pairs by decreasing detection score
    pairs = sorted(pairs, key=lambda x: x[0]['llr'], reverse=True)
    # while the pair list is not empty
    correct_pairs = []
    while pairs:
        # the top pair (system_x, reference_y) is added to the correct detection pair list
        system_x, reference_y = pairs.pop(0)
        correct_pairs.append((system_x, reference_y))
        # remove unused pairs for system_x
        pairs = [pair for pair in pairs if pair[0] != system_x]
        # remove unused pairs for reference_y
        pairs = [pair for pair in pairs if pair[1] != reference_y]

    # get system instances not in correct pairs
    # convert to named tuples to make hashable
    system_misses = []
    if len(system_predictions) > 0:
        SystemTuple = namedtuple('SystemTuple', system_predictions[0])
        system_predictions = [SystemTuple(**x) for x in system_predictions]
        system_misses = set(system_predictions) - set(
            [SystemTuple(**x[0]) for x in correct_pairs])
        # convert back to dictionary
        system_misses = [x._asdict() for x in system_misses]

    # get reference instances not in correct pairs
    # convert to named tuples to make hashable
    reference_misses = []
    if len(reference_data) > 0:
        ReferenceTuple = namedtuple('ReferenceTuple', reference_data[0])
        reference_data = [ReferenceTuple(**x) for x in reference_data]
        reference_misses = set(reference_data) - set(
            [ReferenceTuple(**x[1]) for x in correct_pairs])
        # convert back to dictionary
        reference_misses = [x._asdict() for x in reference_misses]

    return correct_pairs, system_misses, reference_misses


def categorize_pairs(correct_pairs: list[(dict, dict)],
                     system_misses: list[dict], reference_misses: list[dict],
                     threshold: float) -> dict[dict[str, int]]:
    """Categorizes 1) correct pairs, 2) false positives, and 3) false negatives 
    as we vary the detection threshold."""
    # initialize counts
    counts = {
        'correct': 0,
        'false_positive': 0,
        'false_negative': 0,
    }
    # add correct pairs
    counts['correct'] += len(
        [x for x in correct_pairs if x[0]['llr'] >= threshold])
    # add false positives
    counts['false_positive'] += len(
        [x for x in system_misses if x['llr'] >= threshold])
    # add false negatives
    # elements below threshold in correct pairs are false negatives
    counts['false_negative'] += len(
        [x for x in correct_pairs if x[0]['llr'] < threshold])
    # all reference misses are false negatives
    counts['false_negative'] += len(reference_misses)
    return counts


def precision(counts: dict[float, dict[str, int]]) -> float:
    """Calculates precision as the number of correct predictions over the number
    of correct predictions and false positives."""
    # TODO: what to do when precision is undefined?
    # if (counts['correct'] + counts['false_positive']) == 0:
    #     return 0
    return counts['correct'] / (counts['correct'] + counts['false_positive'])


def recall(counts: dict[float, dict[str, int]]) -> float:
    """Calculates recall as the number of correct predictions over the number
    of correct predictions and false negatives."""
    return counts['correct'] / (counts['correct'] + counts['false_negative'])


def compute_precision_recall(
        counts: dict[float, dict[str,
                                 int]]) -> tuple[list[float], list[float]]:
    """Calculates precision and recall for each unique threshold."""
    # calculate precision and recall for each threshold in descending order
    thresholds = sorted(list(set(counts.keys())), reverse=True)
    precision_scores, recall_scores = [], []
    for threshold in thresholds:
        precision_scores.append(precision(counts[threshold]))
        recall_scores.append(recall(counts[threshold]))
    return precision_scores, recall_scores


def average_precision(counts: dict[dict[str, int]]) -> float:
    """Calculates average precision as the "area under the precision-recall
    curve." This is implemented by calculating the precision and recall for each
    unique threshold and then sums the product of precision and recall.
    """
    # calculate precision and recall for each threshold
    ap = 0
    precision_scores, recall_scores = compute_precision_recall(counts)
    # sum the product of precision and recall
    for i in range(len(precision_scores)):
        ap += precision_scores[i] * recall_scores[i]
    return ap


def ap_interp_pr(prec, rec):
    """Return Interpolated P/R curve - Based on VOCdevkit from VOC 2011.
    """
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    return mprec, mrec, idx


def ap_interp(prec, rec):
    """Interpolated AP - Based on VOCdevkit from VOC 2011.
    """
    mprec, mrec, idx = ap_interp_pr(prec, rec)
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap


def calculate_average_precision(system_preds, gold_preds, distances):
    """Amith's implementation of average precision."""
    mapped = map_system_and_gold_predictions(system_preds, gold_preds,
                                             distances)

    xs, ys = [], []
    thresholds = set()
    for file_id in system_preds:
        for system_pred in system_preds[file_id]:
            thresholds.add(system_pred['llr'])
    thresholds = list(sorted(thresholds))

    for threshold in thresholds:
        xs.append(threshold)
        precision, recall = calculate_precision_and_recall(
            system_preds, gold_preds, mapped, threshold)
        ys.append(precision * recall)

    return auc(xs, ys)


def main(args):
    # load data
    system_predictions, anno_dfs, segment_df, versions_df = load_data(
        args.input_dir, args.reference_dir)
    change_point_df = anno_dfs['changepoint.tab']

    # load metadata.csv to filter down to audio and video files
    meta_filepath = os.path.join(
        args.input_dir.split('transformed')[0], 'transformed/metadata.csv')
    meta_df = pd.read_csv(meta_filepath)
    # calculate average precision for each modality
    ap_by_modality = {}
    auc_by_modality = {}
    for modality in meta_df['modality'].unique():
        if modality == 'text':
            threshold = 100
        else:
            threshold = 10
        # filter meta_df to only include modality and mini-eval files
        modality_filter = meta_df['modality'] == modality
        release_filter = meta_df['release'] == 'Mini-Eval'
        file_ids = meta_df[modality_filter & release_filter]['file_uid'].unique()
        # get file_ids from versions_df
        labeled_file_ids = versions_df[versions_df['changepoint_count'] > 0]['file_id'].unique()
        file_ids = list(set(file_ids).intersection(set(labeled_file_ids)))

        # filter change point (label) df to only include labeled modality files
        modality_df = change_point_df[change_point_df['file_id'].isin(file_ids)]

        # map predictions to reference data for each file_id and compute system
        # misses and reference misses
        mappings = {}
        for file_id in file_ids:
            # filter system_predictions down to annotated regions
            # TODO: this can probably be optimized
            preds_df = system_predictions[file_id]
            preds_df['start'] = np.nan
            preds_df['end'] = np.nan
            segments = segment_df[segment_df['file_id'] == file_id][['start', 'end']]
            for i, row in preds_df.iterrows():
                for _, segment in segments.iterrows():
                    if row['timestamp'] >= segment['start'] and row['timestamp'] <= segment['end']:
                        preds_df.loc[i, 'start'] = segment['start']
                        preds_df.loc[i, 'end'] = segment['end']
                        break
            preds_df = preds_df[preds_df['start'].notna()]
            system_predictions[file_id] = preds_df
            system_dict = system_predictions[file_id].to_dict('records')
            reference_dict = modality_df[modality_df['file_id'] ==
                                         file_id].to_dict('records')

            correct_pairs, system_misses, reference_misses = mapping(
                system_dict, reference_dict, threshold)
            mappings[file_id] = {
                'correct_pairs': correct_pairs,
                'system_misses': system_misses,
                'reference_misses': reference_misses,
            }

        # get all unique thresholds across all modality files_ids
        unique_thresholds = set()
        counter = 0
        for file_id in file_ids:
            counter += system_predictions[file_id]['llr'].nunique()
            unique_thresholds.update(
                system_predictions[file_id]['llr'].values.tolist())
        logging.info(f'"{modality}" modality has {counter} unique thresholds.')

        # sort thresholds from highest to lowest
        thresholds = sorted(list(unique_thresholds), reverse=True)

        # # can break thresholds into multiple chunks and compute in parallel
        # num_cores = multiprocessing.cpu_count()
        # chunked_thresholds = np.array_split(thresholds, num_cores)
        # results = Parallel(n_jobs=num_cores)(delayed(compute_precision_recall)(mappings, t) for t in chunked_thresholds)

        # for each threshold, compute the mapping and categorize pairs
        modality_precision_scores, modality_recall_scores = [], []
        for t in tqdm(thresholds):
            threshold_counts = {
                'correct': 0,
                'false_positive': 0,
                'false_negative': 0,
            }
            # map predictions to reference data for each file_id
            for file_id in file_ids:
                file_counts = categorize_pairs(**mappings[file_id],
                                               threshold=t)
                # add file counts to threshold counts
                for key in threshold_counts:
                    threshold_counts[key] += file_counts[key]

            precision_score = precision(threshold_counts)
            recall_score = recall(threshold_counts)
            modality_precision_scores.append(precision_score)
            modality_recall_scores.append(recall_score)

        # calculate average precision for each modality
        # ap_by_modality[modality] = average_precision(threshold_counts)
        ap_by_modality[modality] = ap_interp(modality_precision_scores,
                                             modality_recall_scores)
        auc_by_modality[modality] = auc(modality_recall_scores,
                                        modality_precision_scores)

    # print average precision for each modality
    print('LDC Average Precision by Modality')
    for modality in ap_by_modality:
        print(f'{modality}: {ap_by_modality[modality]:.3f}')
    # print auc for each modality
    print('AUC by Modality')
    for modality in auc_by_modality:
        print(f'{modality}: {auc_by_modality[modality]:.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir',
                        type=str,
                        required=True,
                        help='Directory containing the raw data.')
    parser.add_argument('--reference-dir',
                        type=str,
                        required=True,
                        help='Directory containing the reference data.')
    parser.add_argument('--output-dir',
                        type=str,
                        required=True,
                        help='Directory to write the output to.')
    args = parser.parse_args()
    main(args)