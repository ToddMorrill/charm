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
from utils import DialogAct, check_for_change, get_turn_text


def main():
    # fix random seed so logs are reproducible
    random.seed(10)

    # TODO: set log level from environment variable
    logging.basicConfig(datefmt='%Y-%m-%d %I:%M:%S',
                        format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.DEBUG)
    logging.info('Starting columbia-communication-change')
    if os.environ.get('MODEL_SERVICE') is not None:
        logging.debug(f'Using {os.environ["MODEL_SERVICE"]} backend.')

    asr = CCU.socket(CCU.queues['RESULT'], zmq.SUB)
    result = CCU.socket(CCU.queues['RESULT'], zmq.PUB)

    block_size = 8
    threshold = 0.25

    found_changepoints = []
    all_turns = []
    while True:
        # Start out by waiting for a message on the gesture queue.
        message = CCU.recv_block(asr)

        # Once we get that message, see if it is an ASR message, since
        # that is the only thing we care about.
        if 'type' in message and message['type'] == 'asr_result':
            tic = time.perf_counter()
            turn_text = get_turn_text(message)
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

            # !!!
            # TODO: AWS API call will replace this block

            # if MODEL_SERVICE environment variable set, then query predict endpoint
            if os.environ.get('MODEL_SERVICE') is not None:
                data = {'text':[turn_text]}
                url = f'http://{os.environ["MODEL_SERVICE"]}:8000/predict'
                response = requests.post(url, json=data)
                # response format is [{'label': 'positive', 'score': 0.98}, ..., {}]
                # TODO: response verification
                result = response.json()
                logging.debug(result)
                da.score = result[0]['score']
            else:
                # randomly generate a confidence score for now to simulate model prediction
                da.score = random.random()
            # !!!

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
                CCU.check_message(cp_message)
                CCU.send(result, cp_message)

            toc = time.perf_counter()


if __name__ == "__main__":
    main()