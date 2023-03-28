"""This module runs the main event loop listening for inbound ASR messages and 
calls the communication change model running on AWS.

Examples:
    $ python -u main.py
"""
import logging
import random
import time
import os

import requests
import zmq
from ccu import CCU
from utils import DialogAct, check_for_change, get_turn_text_asr_result_zh, get_turn_text_translation_zh_en


def main():
    # fix random seed so logs are reproducible
    random.seed(10)

    # TODO: set log level from environment variable
    logging.basicConfig(datefmt='%Y-%m-%d %I:%M:%S',
                        format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.DEBUG)
    logging.info('Starting columbia-communication-change')
    if os.environ.get('MODEL_SERVICE') is not None:
        logging.debug(f'Using {os.environ["MODEL_SERVICE"]}:{os.environ["MODEL_PORT"]} backend.')

    asr = CCU.socket(CCU.queues['RESULT'], zmq.SUB)
    result = CCU.socket(CCU.queues['RESULT'], zmq.PUB)

    block_size = 8
    threshold = 0.25

    found_changepoints = []
    all_turns = []

    while True:
        message = CCU.recv_block(asr)

        # Check for ASR Result ("asr_result") Message or 
        # Translation ("translation") Message
        # https://ldc-issues.jira.com/wiki/spaces/CCUC/pages/3060269057/Messages+and+Queues
        if 'type' in message and message['type'] == 'asr_result':
            turn_text = get_turn_text_asr_result_zh(message)
        elif 'type' in message and message['type'] == 'translation':
            turn_text = get_turn_text_translation_zh_en(message)
        else:
            continue

        if not turn_text:
            continue

        # logging.debug(f'Received message: {turn_text}')

        start, end = None, None
        if "start_seconds" in message:
            start = message["start_seconds"]
        if "end_seconds" in message:
            end = message["end_seconds"]

        if len(all_turns) > 0 and start < all_turns[-1].end:
            # print("resetting for new conversation")
            found_changepoints = []
            all_turns = []

        da = DialogAct(turn_text, start, end)

        # if MODEL_SERVICE environment variable set, then query predict endpoint
        if os.environ.get('MODEL_SERVICE') is not None:
            try:
                data = {'text':[turn_text]}
                url = f'http://{os.environ["MODEL_SERVICE"]}:{os.environ["MODEL_PORT"]}/predict'
                response = requests.post(url, json=data)
                # response format is [{'label': 'positive', 'score': 0.98}, ..., {}]
                # TODO: response verification
                json_result = response.json()
                logging.debug(json_result)
                da.score = json_result[0]['score']
            except Exception as e:
                logging.error(f'API error: {e}')
                # randomly generate a confidence score for now to simulate model prediction
                da.score = random.random()    
        else:
            # randomly generate a confidence score for now to simulate model prediction
            da.score = random.random()

        all_turns.append(da)

        change_found = check_for_change(all_turns, found_changepoints,
                                        threshold, block_size)

        if change_found:
            logging.debug('Change point found!')
            cp_message = CCU.base_message('change_point')
            last_cp = found_changepoints[-1]
            # only processing text data, not using timestamp
            cp_message['timestamp'] = last_cp.turn.start
            cp_message['chars'] = int(last_cp.turn.start)
            cp_message['llr'] = last_cp.llr
            cp_message['direction'] = last_cp.tone
            cp_message['container_id'] = "columbia-communication-change"
            # if speaker field present in the message, copy forward
            cp_message['speaker'] = message['speaker'] if 'speaker' in message else 'Unknown'
            
            # if trigger id field in the message, then copy forward and add current message's trigger ID
            # (TODO: add other UUIDs)
            cp_message['trigger_id'] = [message['uuid']]
            if 'trigger_id' in message:
                # ensure trigger_id is a list
                cp_message['trigger_id'] = cp_message['trigger_id'] + list(message['trigger_id'])
            CCU.check_message(cp_message)
            CCU.send(result, cp_message)



if __name__ == "__main__":
    main()