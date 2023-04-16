import os
import sys
import math
import shutil
import pandas as pd
from collections import defaultdict

# sys.path.append('nist_scorer')
# from nist_scorer.CCU_validation_scoring.score_changepoint import score_cp
from CCU_validation_scoring.score_changepoint import score_cp


def filter_system_preds(system_preds, text_char_threshold, time_sec_threshold,
                        filtering):
    if filtering == 'none':
        return system_preds

    assert filtering in {'highest', 'lowest', 'most_similar'}

    by_file = defaultdict(list)
    for system_pred in system_preds:
        by_file[system_pred['file_id']].append(system_pred)

    filtered_system_preds = []
    for file_id, file_system_preds in by_file.items():
        file_system_preds = list(
            sorted(by_file[file_id],
                   key=lambda file_pred: float(file_pred['timestamp'])))

        if len(file_system_preds) <= 1:
            filtered_system_preds.extend(file_system_preds)
            continue

        if file_system_preds[0]['type'] == 'text':
            distance_threshold = text_char_threshold
        else:
            assert file_system_preds[0]['type'] in {'audio', 'video'}
            distance_threshold = time_sec_threshold

        to_remove = set()
        while True:
            candidates = []
            remaining_idxs = list(
                sorted(set(range(len(file_system_preds))) - to_remove))

            if len(remaining_idxs) <= 1:
                break

            for i in range(len(remaining_idxs)):
                distance_before, distance_after = -1, -1

                remaining_idx = remaining_idxs[i]
                if i > 0:
                    before_idx = remaining_idxs[i - 1]
                    distance_before = file_system_preds[remaining_idx][
                        'timestamp'] - file_system_preds[before_idx][
                            'timestamp']
                else:
                    before_idx = None

                if i < len(remaining_idxs) - 1:
                    after_idx = remaining_idxs[i + 1]
                    distance_after = file_system_preds[after_idx][
                        'timestamp'] - file_system_preds[remaining_idx][
                            'timestamp']
                else:
                    after_idx = None

                # if the adjacent predictions are too close, we should consider removing it
                if max(distance_before, distance_after) < distance_threshold:
                    if filtering == 'highest':
                        sort_key = -1 * float(
                            file_system_preds[remaining_idx]['llr'])
                    elif filtering == 'lowest':
                        sort_key = float(
                            file_system_preds[remaining_idx]['llr'])
                    elif filtering == 'most_similar':
                        sort_key = -1 * math.inf
                        if before_idx is not None:
                            sort_key = max(
                                sort_key,
                                abs(
                                    float(file_system_preds[remaining_idx]
                                          ['llr']) -
                                    float(file_system_preds[before_idx]['llr'])
                                ))

                        if after_idx is not None:
                            sort_key = max(
                                sort_key,
                                abs(
                                    float(file_system_preds[remaining_idx]
                                          ['llr']) -
                                    float(file_system_preds[after_idx]['llr']))
                            )
                    else:
                        raise ValueError(
                            f'Unknown filtering type: {filtering}')

                    candidates.append((sort_key, remaining_idx))

            if len(candidates) == 0:
                break

            candidates = list(sorted(candidates))
            to_remove.add(candidates[0][1])

        filtered_system_preds.extend([
            file_system_preds[i] for i in range(len(file_system_preds))
            if i not in to_remove
        ])

    return filtered_system_preds


# refs: array of gold-label LDC references
#   [
#       {
#           'file_id': the LDC file identifier for this changepoint
#           'timestamp': the timestamp of the annotated changepoint
#           'impact_scalar': the impact scalar of the annotated changepoint
#           'type': one of audio / video / text
#       }
#   ]
#   ex:
#       [
#           {'file_id': 'M010015BY', 'timestamp': 1160.2, 'type': 'audio', 'impact_scalar': 4},
#           {'file_id': 'M010015BY', 'timestamp': 1287.6, 'type': 'audio', 'impact_scalar': 2},
#           {'file_id': 'M010029SP', 'timestamp': 288.0, 'type': 'text', 'impact_scalar': 1},
#           {'file_id': 'M010005QD', 'timestamp': 90.2, 'type': 'video', 'impact_scalar': 5},
#           {'file_id': 'M010019QD', 'timestamp': 90, 'type': 'text', 'impact_scalar': 5}
#       ]
# hyps: array of system predictions
#   [
#       {
#           'file_id': the LDC file identifier for this changepoint
#           'timestamp': the timestamp of the annotated changepoint
#           'type': one of audio / video / text
#           'llr': the log-likelihood ratio of the predicted changepoint
#       }
#   ]
#   ex:
#       [
#           {'file_id': 'M010015BY', 'timestamp': 1160.2, 'type': 'audio', 'llr': 1.5},
#           {'file_id': 'M010015BY', 'timestamp': 1287.67, 'type': 'audio', 'llr': 1.5},
#           {'file_id': 'M010029SP', 'timestamp': 288, 'type': 'text', 'llr': 1.5},
#           {'file_id': 'M010005QD', 'timestamp': 90.2, 'llr': 1.5, 'type': 'video'},
#           {'file_id': 'M010019QD', 'timestamp': 190, 'llr': 1.5, 'type': 'text'}
#       ]
# returns a dictionary with an AP score for each document type (audio, video, text)
def calculate_average_precision(refs,
                                hyps,
                                text_char_threshold=100,
                                time_sec_threshold=10,
                                filtering='none'):
    hyps = filter_system_preds(hyps, text_char_threshold, time_sec_threshold,
                               filtering)

    # NIST uses non-zero values of "Class" to indicate annotations / predictions
    # in LDC's randomly selected annotation regions
    for ref in refs:
        ref['Class'] = ref['timestamp']
        ref['start'] = ref['timestamp']
        ref['end'] = ref['timestamp']

    for hyp in hyps:
        hyp['Class'] = hyp['timestamp']
        hyp['start'] = hyp['timestamp']
        hyp['end'] = hyp['timestamp']

    ref_df = pd.DataFrame.from_records(refs)
    hyp_df = pd.DataFrame.from_records(hyps)

    output_dir = 'tmp_scoring_%s' % os.getpid()
    os.makedirs(output_dir, exist_ok=True)

    score_cp(ref_df,
             hyp_df,
             delta_cp_text_thresholds=[text_char_threshold],
             delta_cp_time_thresholds=[time_sec_threshold],
             output_dir=output_dir)

    APs, score_df = {}, pd.read_csv(os.path.join(output_dir,
                                                 'scores_by_class.tab'),
                                    delimiter='\t')
    for _, row in score_df[score_df['metric'] == 'AP'].iterrows():
        APs[row['genre']] = float(row['value'])

    shutil.rmtree(output_dir)

    return APs