import os
import glob
import math
import json
import pickle
import bisect
import getpass
import xmltodict
import pandas as pd
from operator import itemgetter

BASE_DIR = '/mnt/swordfish-pool2/ccu'

SOURCE_DIRECTORIES = {
    'LDC2022E18': [
        'LDC2022E11_CCU_TA1_Mandarin_Chinese_Development_Source_Data_R1',
        'LDC2022E19_CCU_TA1_Mandarin_Chinese_Development_Source_Data_R2_V1.0',
        'LDC2022E19_CCU_TA1_Mandarin_Chinese_Development_Source_Data_R2_V2.0',
        'LDC2022E20_CCU_TA1_Mandarin_Chinese_Development_Source_Data_R3_V1.0',
        'LDC2023E03_CCU_TA1_Mandarin_Chinese_Development_Source_Data_R4_V1.0',
        'LDC2023E06_CCU_TA1_Mandarin_Chinese_Development_Source_Data_R5_V1.0'
    ],
    'LDC2023E01': [
        'LDC2022E22_CCU_TA1_Mandarin_Chinese_Mini_Evaluation_Source_Data'
    ]
}

DATA_TYPES = {
    '.ltf.xml': 'text',
    '.flac.ldcc': 'audio',
    '.mp4': 'video',
    '.mp4.ldcc': 'video'
}

VAL_FILE_IDS = 'loaders/val.json'
TEST_FILE_IDS = 'loaders/test.json'


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
#           'split': the split of the file (train, val, test)
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
#           'utterances': a list of all utterances in the file (only populated if processed=True)
#               {
#                   'start': the starting timestamp of the utterance
#                   'end': the ending timestamp of the utterance
#                   'text': the original language text of the utterance (for audio/video, transcribed by Whisper)
#                   'avg_logprob': only populated for audio/video, produced by Whisper
#                   'no_speech_prob': only populated for audio/video, produced by Whisper
#                   'audio_files': a list of audio files (2-tuples) that span the utterance (length = 0.1 seconds)
#                       (only populated for audio/video if include_preprocessed_audio_and_video=True)
#                       tuple[0] = the starting timestamp of the 0.1 second audio file
#                       tuple[1] = the path to the 0.1 second audio file (relative to 'processed_dir' above)
#                   'video_frames: a list of video frames (2-tuples) that span the utterance
#                       (only populated for audio/video if include_preprocessed_audio_and_video=True)
#                       tuple[0] = the timestamp of the extracted video frame
#                       tuple[1] = the path to the extracted video frame (relative to 'processed_dir' above)
#               }
#       }
def load_ldc_data(include_preprocessed_audio_and_video=False, use_cache=False):
    cache_filepath = os.path.join(BASE_DIR, '%s-cache.pkl' % getpass.getuser())
    if use_cache and os.path.exists(cache_filepath):
        with open(cache_filepath, 'rb') as cache_file:
            return pickle.load(cache_file)

    with open(VAL_FILE_IDS, 'r') as f:
        val_file_ids = set(json.load(f))

    with open(TEST_FILE_IDS, 'r') as f:
        test_file_ids = set(json.load(f))

    file_info_dfs = {
        source_dir: pd.read_csv(os.path.join(BASE_DIR, source_dir, 'docs/file_info.tab'), delimiter='\t')
        for source_dirs in SOURCE_DIRECTORIES.values() for source_dir in source_dirs
    }

    annotated_files = {}
    for segment_file in sorted(set(glob.glob(os.path.join(BASE_DIR, '*/docs/segments.tab')))):
        release_name = segment_file.lstrip(BASE_DIR).split('_')[0]

        # no annotations in this release
        if release_name == 'LDC2022E12':
            continue

        segment_file_df = pd.read_csv(segment_file, delimiter='\t')
        for file_id, rows in segment_file_df.groupby('file_id'):
            start, end = math.inf, -math.inf
            for _, row in rows.iterrows():
                start = min(start, row['start'])
                end = max(end, row['end'])

            data_type, url, status_in_corpora = None, None, []
            for source_dir in SOURCE_DIRECTORIES[release_name]:
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
                    status_in_corpora.append((source_dir, row['status_in_corpus']))

            assert data_type is not None

            if file_id in val_file_ids:
                split = 'val'
                assert file_id not in test_file_ids
            elif file_id in test_file_ids:
                split = 'test'
            else:
                split = 'train'

            file_info = {
                'file_id': file_id,
                'split': split,
                'start': start,
                'end': end,
                'url': url,
                'status_in_corpora': status_in_corpora,
                'data_type': data_type,
                'release': release_name,
                'changepoints': []
            }

            if file_id in annotated_files:
                assert abs(file_info['start'] - annotated_files[file_id]['start']) < 1
                assert abs(file_info['end'] - annotated_files[file_id]['end']) < 1
                assert file_info['url'] == annotated_files[file_id]['url']
                assert file_info['status_in_corpora'] == annotated_files[file_id]['status_in_corpora']
                assert file_info['release'] == annotated_files[file_id]['release']

            annotated_files[file_id] = file_info

        changepoint_file_df = pd.read_csv(
            os.path.join(os.path.dirname(segment_file).rstrip('docs'), 'data/changepoint.tab'), delimiter='\t'
        )
        for file_id, rows in changepoint_file_df.groupby('file_id'):
            changepoints = []
            for _, row in rows.iterrows():
                changepoints.append({
                    'timestamp': float(row['timestamp']),
                    'impact_scalar': row['impact_scalar'],
                    'comment': row['comment'],
                    'annotator': row['user_id']
                })

            changepoints = list(sorted(changepoints, key=lambda changepoint: changepoint['timestamp']))

            assert len(annotated_files[file_id]['changepoints']) == 0

            annotated_files[file_id]['changepoints'].extend(changepoints)

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
        else:
            transcript_files = []
            for (source_dir, _) in file_info['status_in_corpora']:
                if file_info['data_type'] == 'audio':
                    file_name1 = file_id
                else:
                    assert file_info['data_type'] == 'video'
                    file_name1 = '%s-a' % file_id

                transcript_file1 = os.path.join(
                    BASE_DIR, source_dir, 'sources', '%s-transcript.json' % file_name1
                )

                if os.path.exists(transcript_file1):
                    transcript_files.append((file_name1, transcript_file1))

                if file_info['data_type'] == 'video' and file_info['url'] != 'na':
                    if 'bilibili' in file_info['url']:
                        file_name2 = file_info['url'].split('/')[-1]
                        if file_name2.startswith('BV') or file_name2.startswith('av'):
                            file_name2 = file_name2[2:]
                    elif 'youtube' in file_info['url']:
                        file_name2 = file_info['url'].split('=')[-1]
                    else:
                        continue

                    transcript_file2 = os.path.join(
                        BASE_DIR, source_dir, 'sources', '%s-transcript.json' % file_name2
                    )

                    if os.path.exists(transcript_file2):
                        transcript_files.append((file_name2, transcript_file2))

            if len(transcript_files) > 0:
                processed = True

                # we prefer the transcript file that corresponds to the latest release
                file_name, transcript_file = transcript_files[-1]
                with open(transcript_file, 'r') as f:
                    transcript = json.load(f)

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

                if include_preprocessed_audio_and_video:
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

        file_info['processed'] = processed
        file_info['utterances'] = utterances

    if use_cache:
        with open(cache_filepath, 'wb') as f:
            pickle.dump(annotated_files, f)

    return annotated_files
