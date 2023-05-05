import os
import math
import json
import pickle
import bisect
import getpass
import xmltodict
import pandas as pd
from operator import itemgetter

BASE_DIR = '/mnt/swordfish-pool2/ccu'
CORPUS_DIR_PATH = '/mnt/swordfish-pool2/todd/charm/charm/loaders/corpus_dirs.json'
MINI_EVAL_DIR = 'LDC2022E22_CCU_TA1_Mandarin_Chinese_Mini_Evaluation_Source_Data'

with open(CORPUS_DIR_PATH, 'r') as f:
    CORPUS_DIRS = json.load(f)

SOURCE_DIRECTORIES = CORPUS_DIRS['source_data']
SEGMENT_FILES = CORPUS_DIRS['segments_files']

DATA_TYPES = {
    '.ltf.xml': 'text',
    '.flac.ldcc': 'audio',
    '.mp4': 'video',
    '.mp4.ldcc': 'video'
}

ASR_TYPES = {
    'wev2vec2': '_processed_results.json',
    'whisper': '-transcript.json',
    'azure': 'whatever'
}

INTERNAL_VAL_FILE_IDS = '/mnt/swordfish-pool2/todd/charm/charm/loaders/val.json'
INTERNAL_TEST_FILE_IDS = '/mnt/swordfish-pool2/todd/charm/charm/loaders/test.json'

def find_transcript_file(file_info, file_id, asr_type):
    """
        find_transcript_file
    """
    assert asr_type in {'azure', 'wav2vec', 'whisper'}

    transcript_files = []

    for (source_dir, _) in file_info['status_in_corpora']:
        if file_info['data_type'] == 'audio':
            file_name1 = file_id
            file_name1_whisper = file_id
        else:
            assert file_info['data_type'] == 'video'
            file_name1_whisper = '%s-a' % file_id
            file_name1 = file_id

        if asr_type == 'azure':
            transcript_file1 = os.path.join(
                BASE_DIR, source_dir, 'sources', f"{asr_type}/{file_name1}.json"
            )

            if os.path.exists(transcript_file1):
                transcript_files.append((file_name1, transcript_file1))
            else:
                transcript_file1 = os.path.join(
                    BASE_DIR, MINI_EVAL_DIR, 'sources', f"{asr_type}/{file_name1}.json"
                )
                if os.path.exists(transcript_file1):
                    transcript_files.append((file_name1, transcript_file1))
                else:
                    print(f"Can't find {file_name1} for azure")

        elif asr_type == 'wav2vec':
            transcript_file1 = os.path.join(
                BASE_DIR, source_dir, 'sources', f"{asr_type}/{file_name1}_processed_results.json"
            )

            if os.path.exists(transcript_file1):
                transcript_files.append((file_name1, transcript_file1))
            else:
                transcript_file1 = os.path.join(
                    BASE_DIR, MINI_EVAL_DIR, 'sources', f"{asr_type}/{file_name1}_processed_results.json"
                )
                if os.path.exists(transcript_file1):
                    transcript_files.append((file_name1, transcript_file1))
                else:
                    print(f"Can't find {file_name1} for wav2vec")

        else:
            if 'EVALUATION_LDC2023E07' in file_info['splits']:
                source_dir = 'LDC2023E07_CCU_TA1_Mandarin_Chinese_Evaluation_Source_Data_V1.0'
            transcript_file1 = os.path.join(
                BASE_DIR, source_dir, 'sources', f"{file_name1_whisper}-transcript.json"
            )

            if os.path.exists(transcript_file1):
                transcript_files.append((file_name1_whisper, transcript_file1))

        if file_info['data_type'] == 'video' and file_info['url'] != 'na':
            if 'bilibili' in file_info['url']:
                file_name2 = file_info['url'].split('/')[-1]
                if file_name2.startswith('BV') or file_name2.startswith('av'):
                    file_name2 = file_name2[2:]
            elif 'youtube' in file_info['url']:
                file_name2 = file_info['url'].split('=')[-1]
            else:
                continue

            if asr_type == 'whisper':
                transcript_file2 = os.path.join(
                    BASE_DIR, source_dir, 'sources', f"{file_name2}-transcript.json"
                )
            else:
                transcript_file2 = os.path.join(
                    BASE_DIR, source_dir, f"{file_name2}{ASR_TYPES[asr_type]}"
                )

            if os.path.exists(transcript_file2):
                transcript_files.append((file_name2, transcript_file2))

    return transcript_files


def utterances_from_transcript(transcript, transcript_file, utterances, include, file_info, file_name):

    # Segment processing
    if 'segments' in transcript and file_name != 'M010050LM-a':
        for segment in transcript['segments']:
            utterances.append({
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'],
                'avg_logprob': segment['avg_logprob'],
                'no_speech_prob': segment['no_speech_prob'],
                'audio_files': [],
                'video_frames': []
            })
    else:
        if transcript:
            if 'asr_turn_lvl' in transcript and transcript['asr_turn_lvl']:
                for segment in transcript['asr_turn_lvl']:
                    utterances.append({
                        'start': segment['start_time'],
                        'end': segment['end_time'],
                        'text': segment['transcript'],
                        'audio_files': [],
                        'video_frames': []
                    })
            elif 'asr_utterance_lvl' in transcript and transcript['asr_utterance_lvl']:
                for segment in transcript['asr_utterance_lvl']:
                    utterances.append({
                        'start': segment['start_time'],
                        'end': segment['end_time'],
                        'text': segment['transcript'],
                        'audio_files': [],
                        'video_frames': []
                    })
        else:
            return

    if include:
        processed_dir = os.path.join(
            os.path.dirname(transcript_file).replace('sources', 'processed'), file_name
        )
        processed_files = os.listdir(processed_dir)

        file_info['processed_dir'] = processed_dir
        add_processed_frames(
            '.wav', 'audio_files', processed_files, utterances
        )

        if file_info['data_type'] == 'video':
            add_processed_frames(
                '.jpg', 'video_frames', processed_files, utterances
            )


def add_processed_frames(file_ext, update_key, processed_files, utterances):
    matching_files = list(sorted(
        [
            (
                *tuple(map(float, os.path.basename(processed_file).split('-')[:3])), processed_file
            ) for processed_file in processed_files if processed_file.endswith(file_ext)
        ]
    ))
    matching_file_timestamps = [
        60 * 60 * hours + 60 * minutes + seconds for hours, minutes, seconds, _ in matching_files
    ]
    assert len(matching_files) > 0

    for utterance in utterances:
        start, end = utterance['start'], utterance['end']
        # first file with timestamp >= start
        first_file_idx = bisect.bisect_left(matching_file_timestamps, start)
        # first file with timestamp > end
        last_file_idx = bisect.bisect_right(matching_file_timestamps, end, lo=first_file_idx)
        utterance[update_key].extend(
            zip(
                matching_file_timestamps[first_file_idx:last_file_idx],
                map(itemgetter(3), matching_files[first_file_idx:last_file_idx])
            )
        )


# returns a dictionary that aggregates the annotations for each file in the LDC releases
#   key: file_id
#   value: a dictionary of file information
#       {
#           'file_id': the LDC identifier of the file
#           'splits': a set indicate the splits which contain this file {INTERNAL_TRAIN, INTERNAL_VAL, INTERNAL_TEST, EVALUATION_LDC2023E07}
#           'processed': whether or not the file has been successfully downloaded and processed
#               (ignore all files where processed = False)
#           'start': the starting timestamp of the annotated region in the file
#           'end': the ending timestamp of the annotated region in the file
#           'data_type': the original data type of the file (text, audio, video)
#           'processed_dir': the directory which contains processed audio files and video frames
#           'changepoints': a list of changepoints LDC identified in the file's annotated region (start <= t <= end)
#               {
#                   'timestamp': the timestamp of the annotated changepoint
#                   'impact_scalar': the impact scalar of the annotated changepoint
#                   'comment': the annotator's explanation for the annotated changepoint
#                   'annotator': the LDC identifier of the annotator
#               }
#           'utterances':
#               # if data_type=text, a list of of all utterances in the file
#               # if data_type=audio/video, a dictionary of lists of all utterances in the file (only populated if processed=True)
#                   key: one of whisper, wav2vec, azure (the system used to transcribe the utterances)
#                   [{
#                       'start': the starting timestamp of the utterance
#                       'end': the ending timestamp of the utterance
#                       'text': the original language text of the utterance (for audio/video, transcribed by identified system)
#                       'avg_logprob': only populated for audio/video, only populated for Whisper
#                       'no_speech_prob': only populated for audio/video, only populated for Whisper
#                       'audio_files': a list of audio files (2-tuples) that span the utterance (length = 0.1 seconds)
#                           (only populated for audio/video if include_preprocessed_audio_and_video=True)
#                           tuple[0] = the starting timestamp of the 0.1 second audio file
#                           tuple[1] = the path to the 0.1 second audio file (relative to 'processed_dir' above)
#                       'video_frames: a list of video frames (2-tuples) that span the utterance
#                           (only populated for audio/video if include_preprocessed_audio_and_video=True)
#                           tuple[0] = the timestamp of the extracted video frame
#                           tuple[1] = the path to the extracted video frame (relative to 'processed_dir' above)
#                   }]
#       }

def load_ldc_data(include_preprocessed_audio_and_video=False, use_cache=False):
    cache_filepath = os.path.join(BASE_DIR, '%s-cache.pkl' % getpass.getuser())

    if use_cache and os.path.exists(cache_filepath):
        with open(cache_filepath, 'rb') as cache_file:
            return pickle.load(cache_file)

    with open(INTERNAL_VAL_FILE_IDS, 'r') as f:
        internal_val_file_ids = set(json.load(f))

    with open(INTERNAL_TEST_FILE_IDS, 'r') as f:
        internal_test_file_ids = set(json.load(f))

    annotated_files = {}
    for data_subset in ['dev_cn', 'test_cn']:
        file_info_dfs = {
            source_dir: pd.read_csv(os.path.join(BASE_DIR, source_dir, 'docs/file_info.tab'), delimiter='\t')
            for source_dirs in SOURCE_DIRECTORIES[data_subset].values() for source_dir in source_dirs
        }

        for segment_file in CORPUS_DIRS["segments_files"][data_subset]:
            segment_file = os.path.join(BASE_DIR, segment_file)
            release_name = segment_file.lstrip(BASE_DIR).split('_')[0]

            # no data files in this release
            if release_name == 'LDC2022E12':
                continue

            segment_file_df = pd.read_csv(segment_file, delimiter='\t')
            file_id_column = 'file_id' if data_subset == 'dev_cn' else 'file_uid'

            for file_id, rows in segment_file_df.groupby(file_id_column):
                start, end = math.inf, -math.inf

                if data_subset == 'dev_cn':
                    for _, row in rows.iterrows():
                        start = min(start, row['start'])
                        end = max(end, row['end'])
                else:
                    assert data_subset == 'test_cn'
                    start = 0.0
                    end = rows.iloc[0]["length"]  # only duplicates when dealing with txt format

                data_type, url, status_in_corpora = None, None, []

                for source_dir in SOURCE_DIRECTORIES[data_subset][release_name]:
                    file_info_df = file_info_dfs[source_dir]
                    for _, row in file_info_df[file_info_df['file_uid'] == file_id].iterrows():
                        if row['data_type'] == '.psm.xml':
                            continue

                        if data_type is not None:
                            assert data_type == DATA_TYPES[row['data_type']]

                        data_type = DATA_TYPES[row['data_type']]

                        if url is not None:
                            assert url == row['url']

                        url = row['url']
                        if data_subset == 'test_cn' and 'text' != data_type:
                            status_in_corpora.append(
                                (
                                    'LDC2023E07_CCU_TA1_Mandarin_Chinese_Evaluation_Source_Data_V1.0',
                                    row['status_in_corpus']
                                )
                            )
                        elif data_subset == 'test_cn' and row['file_path'].split("/")[0] != 'data':
                            status_in_corpora.append((row['file_path'].split("/")[0], row['status_in_corpus']))
                        else:
                            status_in_corpora.append((source_dir, row['status_in_corpus']))

                assert data_type is not None

                if data_subset == 'dev_cn':
                    if file_id in internal_val_file_ids:
                        split = 'INTERNAL_VAL'
                        assert file_id not in internal_test_file_ids
                    elif file_id in internal_test_file_ids:
                        split = 'INTERNAL_TEST'
                    else:
                        split = 'INTERNAL_TRAIN'
                else:
                    assert data_subset == 'test_cn'
                    split = "EVALUATION_LDC2023E07"

                file_info = {
                    'file_id': file_id,
                    'splits': {split},
                    'start': start,
                    'end': end,
                    'url': url,
                    'status_in_corpora': status_in_corpora,
                    'data_type': data_type,
                    'release': release_name,
                    'changepoints': []
                }

                # we skip test_cn as the eval corpus includes the minieval corpus
                if file_id in annotated_files:
                    if data_subset != 'test_cn':
                        assert abs(file_info['start'] - annotated_files[file_id]['start']) < 1
                        assert abs(file_info['end'] - annotated_files[file_id]['end']) < 1
                        assert file_info['url'] == annotated_files[file_id]['url']
                        assert file_info['status_in_corpora'] == annotated_files[file_id]['status_in_corpora']
                        assert file_info['release'] == annotated_files[file_id]['release']
                    annotated_files[file_id]['splits'].add(split)
                else:
                    annotated_files[file_id] = file_info

            # For test set, we don't have changepoints a.s. we don't need to trigger this.
            if data_subset == 'dev_cn':
                changepoint_file_df = pd.read_csv(
                    os.path.join(os.path.dirname(segment_file).rstrip('docs'), 'data/changepoint.tab'), delimiter='\t'
                )
                for file_id, rows in changepoint_file_df.groupby('file_id'):
                    changepoints = []
                    for _, row in rows.iterrows():
                        changepoints.append({
                            'timestamp': row['timestamp'],
                            'impact_scalar': row['impact_scalar'],
                            'comment': row['comment'],
                            'annotator': row['user_id']
                        })

                    changepoints = list(sorted(changepoints, key=lambda changepoint: changepoint['timestamp']))

                    if len(annotated_files[file_id]['changepoints']) > 0:
                        assert len(annotated_files[file_id]['changepoints']) == len(changepoints)
                        for old_cp, new_cp in zip(annotated_files[file_id]['changepoints'], changepoints):
                            assert old_cp['timestamp'] == new_cp['timestamp']
                    else:
                        annotated_files[file_id]['changepoints'].extend(changepoints)

    # Common for INTERNAL_TRAIN/VAL/TEST and EVALUATION
    for file_id in sorted(annotated_files.keys()):
        file_info = annotated_files[file_id]

        processed, utterances = False, []
        if file_info['data_type'] == 'text':
            processed = True
            # we prefer the file that corresponds to the latest release
            source_dir, _ = file_info['status_in_corpora'][-1]
            ltf_filepath = os.path.join(
                BASE_DIR, source_dir, 'data/text/ltf/%s.ltf.xml' % file_id
            )
            with open(ltf_filepath, 'r') as f:
                ltf_text = xmltodict.parse(f.read())
                for segment in ltf_text['LCTL_TEXT']['DOC']['TEXT']['SEG']:
                    utterances.append(
                        {
                            'start': int(segment['@start_char']),
                            'end': int(segment['@end_char']),
                            'text': segment['ORIGINAL_TEXT'],
                            'audio_files': [],
                            'video_frames': []
                        }
                    )
                file_info['utterances'] = utterances
        else:
            # FIND Transcript Files
            transcript_files = {
                'whisper': [],
                'wav2vec': [],
                'azure': []
            }
            # Default - whisper
            transcript_files['whisper'] = find_transcript_file(file_info, file_id, 'whisper')

            if 'EVALUATION_LDC2023E07' in file_info["splits"]:
                # Break if we don't find a transcript ~~ should never happen.
                transcript_files['wav2vec'] = find_transcript_file(file_info, file_id, 'wav2vec')
                assert len(transcript_files['wav2vec']) > 0
                transcript_files['azure'] = find_transcript_file(file_info, file_id, 'azure')
                assert len(transcript_files['azure']) > 0

            # for video / audio only
            if len(transcript_files['whisper']) > 0:
                processed = True

                if 'EVALUATION_LDC2023E07' in file_info['splits']:
                    utterances_wav2vec, utterances_azure = [], []

                    file_name, whisper_transcript_f = transcript_files['whisper'][-1]

                    _, wav2vec_transcript_f = transcript_files['wav2vec'][-1]
                    _, azure_transcript_f = transcript_files['azure'][-1]

                    with open(whisper_transcript_f, 'r') as f:
                        whisper_transcript = json.load(f)
                    with open(wav2vec_transcript_f, 'r') as f:
                        wav2vec_transcript = json.load(f)
                    with open(azure_transcript_f, 'r') as f:
                        azure_transcript = json.load(f)

                    utterances_from_transcript(
                        whisper_transcript, whisper_transcript_f, utterances,
                        include_preprocessed_audio_and_video, file_info, file_name
                    )

                    utterances_from_transcript(
                        wav2vec_transcript, whisper_transcript_f, utterances_wav2vec,
                        include_preprocessed_audio_and_video, file_info, file_name
                    )

                    utterances_from_transcript(
                        azure_transcript, whisper_transcript_f, utterances_azure,
                        include_preprocessed_audio_and_video, file_info, file_name
                    )

                    file_info['utterances'] = {
                        'whisper': utterances,
                        'wav2vec': utterances_wav2vec,
                        'azure': utterances_azure
                    }
                else:
                    # we prefer the transcript file that corresponds to the latest release
                    file_name, transcript_file = transcript_files['whisper'][-1]

                    with open(transcript_file, 'r') as f:
                        transcript = json.load(f)

                    utterances_from_transcript(
                        transcript, transcript_file, utterances,
                        include_preprocessed_audio_and_video, file_info, file_name
                    )

                    file_info['utterances'] = {
                        'whisper': utterances
                    }
            else:
                file_info['utterances'] = {}
        file_info['processed'] = processed

    count_eval_files = 0
    count_dev_files = 0

    # Last check to see we have all the files:
    for file_key in annotated_files:
        file_info = annotated_files[file_key]
        if 'EVALUATION_LDC2023E07' in file_info['splits']:
            count_eval_files += 1

            if file_info['data_type'] in {'audio', 'video'}:
                assert len(file_info['utterances']['whisper']) > 0
                assert len(file_info['utterances']['wav2vec']) > 0
                if 'M01004G0B' not in file_info['file_id']:
                    assert len(file_info['utterances']['azure']) > 0
        else:
            if file_info['utterances']:
                count_dev_files += 1

    assert (count_eval_files == 6_666)

    if use_cache:
        with open(cache_filepath, 'wb') as f:
            pickle.dump(annotated_files, f)

    return annotated_files


if __name__ == '__main__':
    exit(load_ldc_data(False, True))
