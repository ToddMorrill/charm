"""This file takes as input an LDC data directory and produces jsonl files for
the transcripts of these files.

Examples:
    $ python create_jsonl.py --data-dir ~/Documents/data/charm/raw/LDC2022E11_CCU_TA1_Mandarin_Chinese_Development_Source_Data_R1
"""
import argparse
import json
import datetime
import os
import uuid

import pandas as pd
import whisper
from ccu import CCU
from utils import CCUDataLoader, TextData


def strip_ldc_header(input_filepath):
    os.makedirs('./transcripts', exist_ok=True)
    # create output filename
    # TODO: add option to save this file elsewhere
    out_file = os.path.basename(input_filepath)
    out_file = out_file.split('.ldcc')[0]
    out_filepath = os.path.join('./transcripts', out_file)
    with open(input_filepath, 'rb') as input:
        with open(out_filepath, 'wb') as output:
            output.write(input.read()[1024:])
    return out_filepath


def write_json(filepath, dict_object):
    with open(filepath, 'w') as f:
        json.dump(dict_object, f, ensure_ascii=False)


def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def create_messages(text_data):
    messages = []
    ctr = 1
    for turn in text_data:
        message = {
            "queue": "RESULT",
            "time_seconds": ctr,
            "message": {
                "type": "asr_result",
                "uuid": turn.uuid,
                "start_seconds": turn.start,
                "end_seconds": turn.end,
                "asr_text": turn.text,
                "asr_type": "CONSOLIDATED_RESULT",
                "datetime": str(datetime.datetime.now())
            }
        }
        messages.append(message)
        ctr += 1
    return messages


def create_text_objects(segments):
    text_objects = []
    for idx, seg in enumerate(segments):
        start = seg['start']
        end = seg['end']
        text = seg['text']
        uuid = uuid.uuid4()
        text_objects.append(TextData(str(idx), start, end, text, uuid))
    return text_objects

def main(args):
    data_dir = args.data_dir
    if os.path.basename(data_dir) != 'data':
        data_dir = os.path.join(data_dir, 'data')

    loader = CCUDataLoader(data_dir)
    # load list of files
    release_dir, _ = os.path.split(data_dir)
    file_info_df = pd.read_csv(os.path.join(release_dir, 'docs',
                                            'file_info.tab'),
                               delimiter='\t')

    # filter for a text, audio, and video file
    text_file_id = file_info_df[file_info_df['data_type'] ==
                                '.ltf.xml'].iloc[0]['file_uid']
    text_transcript = loader.load_text(text_file_id)

    video_file_id = file_info_df[file_info_df['data_type'] ==
                                 '.mp4.ldcc'].iloc[0]['file_uid']
    in_video_file_path = os.path.join(data_dir, 'video',
                                      f'{video_file_id}.mp4.ldcc')

    audio_file_id = file_info_df[file_info_df['data_type'] ==
                                 '.flac.ldcc'].iloc[0]['file_uid']
    in_audio_file_path = os.path.join(data_dir, 'audio',
                                      f'{audio_file_id}.flac.ldcc')

    video_transcript = os.path.join('./transcripts', f'{video_file_id}.json')
    audio_transcript = os.path.join('./transcripts', f'{audio_file_id}.json')
    transcripts = {}
    for modality, input_filepath, transcript in [
        ('video', in_video_file_path, video_transcript),
        ('audio', in_audio_file_path, audio_transcript)
    ]:
        if os.path.exists(transcript):
            transcripts[modality] = load_json(transcript)
            continue

        # strip LDC headers
        clean_filepath = strip_ldc_header(input_filepath)

        # generate one audio and one video transcript
        model = whisper.load_model('large', device='cuda')
        # breakpoint()
        decode_options = {'language': 'Chinese'}
        result = model.transcribe(clean_filepath, **decode_options)
        write_json(transcript, result)

    # TODO: pick up here
    doc_name = docs_type + "_" + doc

    filename = "new_data/{}.jsonl".format(doc_name)
    with open(filename, 'w') as out:
        for ddict in messages:
            jout = json.dumps(ddict) + '\n'
            out.write(jout)

    filename = "short_data/{}.jsonl".format(doc_name)
    with open(filename, 'w') as out:
        for ddict in messages[:60]:
            jout = json.dumps(ddict) + '\n'
            out.write(jout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',
                        help='Directory for a particular LDC data release',
                        required=True)
    args = parser.parse_args()
    main(args)