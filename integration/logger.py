#
# Very simple logging function. 
# TODO: spins way too much.
#

import argparse
from datetime import datetime
import json
import logging

import zmq

from ccu import CCU

def make_human_readable(message):
    """Trims any string field which is longer than max_len characters (usually 60)
       down to 60 characters and adds a ... if it does so."""
    max_len = 60
    for key, value in message.items():
        if type(value) == str and len(value) > max_len:
            message[key] = value[:max_len] + " ..."
    return message 

logging.basicConfig(datefmt='%Y-%m-%d %I:%M:%S',
                    format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG)

parser = argparse.ArgumentParser('Program to send many messages for testing.')
parser.add_argument('-c', '--complete', default=False, action='store_true',
                    # Python 3.9: action=argparse.BooleanOptionalAction, 
                    help = 'Print out entire fields, even very long ones.')
parser.add_argument('-j', '--jsonl', type=str, 
                    help = 'Save output to a JSONL format file for later use by script.py.')
parser.add_argument('-p', '--pretty', default=False, action='store_true',
                    # Python 3.9: action=argparse.BooleanOptionalAction, 
                    help = 'Pretty print output: json messages will be more understandable, but multiline.')

parser.parse_args()
args = parser.parse_args()

for name, queue in CCU.queues.items():
    queue["socket"] = CCU.socket(queue, zmq.SUB)
    
logging.info(f'CCU Logger Running for version {CCU.__version__} on {CCU.host_ip}.')
logging.info(f'with config file {CCU.config_file_path}.')   

if args.jsonl is not None:
    json_file = open(args.jsonl,'a')
    json_file.write(f'# JSONL saved from logger.py at {datetime.now()}\n')
    
start_time = None

while True:
    # For each queue, see if there is a message, and print if there is
    # (with the current time).
    for name, queue in CCU.queues.items():
        message = CCU.recv(queue["socket"])
        if message is not None:
            if args.jsonl is not None:
                wrapped = {}
                if start_time is None:
                    start_time = datetime.now()
                    wrapped['time_seconds'] = 0.0
                else:
                    delta = datetime.now() - start_time
                    # Next line converts to floating point
                    wrapped['time_seconds'] = delta.total_seconds()
                wrapped['queue'] = name
                wrapped['message'] = message
                json_file.write(json.dumps(wrapped)+'\n') #, separators=(',', ':')))
                json_file.flush()
            if args.pretty:
                print(f'{CCU.now()} on {name}: ')
                if not args.complete:
                    make_human_readable(message)
                human_readable = json.dumps(message, indent=4)
                print(human_readable)
            else:
                if not args.complete:
                    make_human_readable(message)
                print(f'{CCU.now()} on {name}: {message}')