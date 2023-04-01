"""This file takes as input an LDC data directory and produces jsonl files for
the transcripts of these files. Optionally creates synthetic test for translation messages by injecting wrong data randomly to test robustness.

Examples:
    $ python create_jsonl.py --data-dir ~/Documents/data/charm/raw/LDC2022E11_CCU_TA1_Mandarin_Chinese_Development_Source_Data_R1
    $ python create_jsonl.py --data-dir ~/Documents/data/charm/raw/LDC2022E11_CCU_TA1_Mandarin_Chinese_Development_Source_Data_R1 --translations
"""
import argparse
import json
import datetime
import os
import uuid
import logging

import pandas as pd
import whisper
from ccu import CCU
from utils import CCUDataLoader, TextData

logging.basicConfig(datefmt='%Y-%m-%d %I:%M:%S',
                        format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.DEBUG)

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

def create_translation_messages(text_data,):
    messages = []
    ctr = 1
    for i, turn in enumerate(text_data):
        if i % 2 == 0:
            translation_message = {
                    "asr_type":"TURN",
                    "source_language_code":"en",
                    "trigger_id":["NA"],
                    "source_language":"English",
                    "target_language":"Mandarin Chinese",
                    "target_language_code": "zh",
                    "type":"translation",
                    "uuid":turn.uuid,
                    "audio_id":"NA",
                    "datetime":str(datetime.datetime.now()),
                    "translation_romanized":"NA",
                    "container_name":"columbia-communication-change",
                    "start_seconds":turn.start,
                    "end_seconds":turn.end,
                    "engine": "MarianNMT",
                    "vendor": "OpenAI",
                    "speaker": "FLE",
                    "translation":turn.text,
                    "text":turn.text,
                    "segment_id":"NA",
                    "audio_source":"AUDIO_ENV"
            }
            asr_message = {
                "type": "asr_result",
                "uuid": turn.uuid,
                "start_seconds": turn.start,
                "end_seconds": turn.end,
                "asr_text": "[ENGLISH TEXT INPUT]",
                "asr_type": "TURN",
                "datetime": str(datetime.datetime.now()),
                "container_name": "columbia-communication-change",
                "trigger_id": ["NA"],
                "vendor": "OpenAI",
                "engine": "Whisper",
                "audio_source": "AUDIO_ENV",
                "audio_id": "NA",
                "segment_id": "NA",
                "asr_json": "NA",
                "asr_language": "English",
                "asr_language_code": "en",
                "speaker": "FLE",
            }
        else:
            asr_message = {
                "type": "asr_result",
                "uuid": turn.uuid,
                "start_seconds": turn.start,
                "end_seconds": turn.end,
                "asr_text": turn.text,
                "asr_type": "TURN",
                "datetime": str(datetime.datetime.now()),
                "container_name": "columbia-communication-change",
                "trigger_id": ["NA"],
                "vendor": "OpenAI",
                "engine": "Whisper",
                "audio_source": "AUDIO_ENV",
                "audio_id": "NA",
                "segment_id": "NA",
                "asr_json": "NA",
                "asr_language": "Chinese",
                "asr_language_code": "zh",
                "speaker": "FLE",
            }
            translation_message = {
                    "asr_type":"TURN",
                    "source_language_code":"zh",
                    "trigger_id":["NA"],
                    "source_language":"Mandarin Chinese",
                    "target_language":"English",
                    "target_language_code": "en",
                    "type":"translation",
                    "uuid":turn.uuid,
                    "audio_id":"NA",
                    "datetime":str(datetime.datetime.now()),
                    "translation_romanized":"NA",
                    "container_name":"columbia-communication-change",
                    "start_seconds":turn.start,
                    "end_seconds":turn.end,
                    "engine": "MarianNMT",
                    "vendor": "OpenAI",
                    "speaker": "FLE",
                    "translation":"[ENGLISH TRANSLATED TEXT]",
                    "text":turn.text,
                    "segment_id":"NA",
                    "audio_source":"AUDIO_ENV"
            }
        
        message_translation = {
            "queue": "RESULT",
            "time_seconds": ctr,
            "message": translation_message
        }
        
        CCU.check_message(translation_message)

        message_asr = {
            "queue": "RESULT",
            "time_seconds": ctr,
            "message": asr_message
        }

        CCU.check_message(asr_message)
        messages.append(message_asr)
        messages.append(message_translation)
        ctr += 1
    return messages

def create_messages(text_data, ):
    messages = []
    ctr = 1
    for turn in text_data:
        embedded_message = {
                "type": "asr_result",
                "uuid": turn.uuid,
                "start_seconds": turn.start,
                "end_seconds": turn.end,
                "asr_text": turn.text,
                "asr_type": "TURN",
                "datetime": str(datetime.datetime.now()),
                "container_name": "columbia-communication-change",
                "trigger_id": ["NA"],
                "vendor": "OpenAI",
                "engine": "Whisper",
                "audio_source": "AUDIO_ENV",
                "audio_id": "NA",
                "segment_id": "NA",
                "asr_json": "NA",
                "asr_language": "Chinese",
                "asr_language_code": "zh",
                "speaker": "FLE",
            }
        message = {
            "queue": "RESULT",
            "time_seconds": ctr,
            "message": embedded_message
        }
        CCU.check_message(embedded_message)
        messages.append(message)
        ctr += 1
    return messages


def create_text_objects(segments):
    text_objects = []
    for idx, seg in enumerate(segments):
        start = seg['start']
        end = seg['end']
        text = seg['text']
        text_objects.append(TextData(str(idx), start, end, text, str(uuid.uuid4())))
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

    if args.translations:
        # if testing translation capabilities, select the 1st index from each modality
        idx = 1
    else:
        # default to selecting the 0th index from each modality
        idx = 0

    # filter for a text, audio, and video file
    text_file_id = file_info_df[file_info_df['data_type'] ==
                                '.ltf.xml'].iloc[idx]['file_uid']
    logging.info(f'Loading text file {text_file_id}...')
    text_transcript = loader.load_text(text_file_id)

    video_file_id = file_info_df[file_info_df['data_type'] ==
                                 '.mp4.ldcc'].iloc[idx]['file_uid']
    in_video_file_path = os.path.join(data_dir, 'video',
                                      f'{video_file_id}.mp4.ldcc')

    audio_file_id = file_info_df[file_info_df['data_type'] ==
                                 '.flac.ldcc'].iloc[idx]['file_uid']
    in_audio_file_path = os.path.join(data_dir, 'audio',
                                      f'{audio_file_id}.flac.ldcc')

    video_transcript = os.path.join('./transcripts', f'{video_file_id}.json')
    audio_transcript = os.path.join('./transcripts', f'{audio_file_id}.json')
    transcripts = {}
    for modality, input_filepath, transcript in [
        ('video', in_video_file_path, video_transcript),
        ('audio', in_audio_file_path, audio_transcript)
    ]:
        logging.info(f'Getting transcript for {modality} file {input_filepath}...')
        # print(transcript)
        if os.path.exists(transcript):
            logging.info('Transcript already exists, loading...')
            transcripts[modality] = load_json(transcript)
            continue

        # strip LDC headers
        clean_filepath = strip_ldc_header(input_filepath)
        
        # generate transcripts
        if 'model' not in locals(): # only load model once
            model = whisper.load_model('large', device='cuda')
        decode_options = {'language': 'Chinese'}
        result = model.transcribe(clean_filepath, **decode_options)
        write_json(transcript, result)

    # convert to standardized format
    for modality in transcripts:
        transcripts[modality] = create_text_objects(transcripts[modality]['segments'])
        if args.translations:
            transcripts[modality] = create_translation_messages(transcripts[modality])
        else:
            transcripts[modality] = create_messages(transcripts[modality])
    
    if args.translations:
        transcripts['text'] = create_translation_messages(text_transcript)
    else:
        transcripts['text'] = create_messages(text_transcript)

    file_id_map = {'text': text_file_id, 'video': video_file_id, 'audio': audio_file_id}
    for modality in transcripts:
        if args.translations:
            file_name = f'{modality}_{file_id_map[modality]}_input_translation.jsonl'
        else:
            file_name = f'{modality}_{file_id_map[modality]}_input.jsonl'
        filepath = os.path.join('./transcripts', file_name)
        # write jsonl format
        with open(filepath, 'w') as out:
            for message in transcripts[modality]:
                jout = json.dumps(message, ensure_ascii=False) + '\n'
                out.write(jout)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',
                        help='Directory for a particular LDC data release',
                        required=True)
    parser.add_argument('--translations', action='store_true', default=False, help='Generate translation messages if flag is passed.')
    args = parser.parse_args()
    main(args)