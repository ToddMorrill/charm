#! /usr/bin/env python
# This script filters jsonl logs in various ways.

from argparse import RawDescriptionHelpFormatter
import argparse
import base64
from ccu import CCU
import json
import logging
import sys
import time

sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')

parser = argparse.ArgumentParser( 
        formatter_class=RawDescriptionHelpFormatter,
        epilog='\nProgram to check messages in jsonl files.\nExample command lines:\n'+
        '  Check a jsonl file and print out errors in a human readable, message by message format:\n'+
        '    ./jsonlcheck.py t2.jsonl\n'+
        '  Check a jsonl file and print out errors in a line by line, logger friendly format:\n'+
        '    ./jsonlcheck.py -l t1.jsonl\n')
parser.add_argument('-c', '--containers', type=str,
                    help = 'Names of containers you want to check, seperated by commas.  For example: --containers sri-whisper will only print out messages from that one container, while -c sri-whisper,columbia-norms will print out message from those two containers.  Do not put spaces around the commas.  Names must match exactly without any kind of wildcard characters or partial matching.  Queue filtering (the -q option) is done first.')
parser.add_argument('-e', '--exclude', action='store_true',
                    help = 'Exclude queues or messages from the output.  Used with the -q and -m options.')
parser.add_argument('infile', type=argparse.FileType('r', encoding='utf-8'), nargs='?', default=sys.stdin,
                    help = 'The file to use.  Stdin is used if no file is given.')
parser.add_argument('-l', '--line', action='store_false', dest='block_format', default=True,
                    help = 'Print out errors in line/logger format.')
parser.add_argument('-m', '--messages', type=str,
                    help = 'Names of messasges you want to log, seperated by commas.  For example: --messasges valence will print out valence messages only, while --m valence,norm_occurrence,hello will print out those three message types.  Do not put spaces around the commas.  Names must match exactly without any kind of wildcard characters or partial matching.  Queue filtering (the -q option) is done first.')
parser.add_argument('-n', '--no_error_messages', action='store_true',
                    help = 'Do not print error messages related to message formatting or fields.')
parser.add_argument('-p', '--provenance', action='store_true',
                    help = 'Only check for messsage provenance, not anything else.')
parser.add_argument('-q', '--queues', type=str,
                    help = 'Names of queues you want to log, seperated by commas.  For example: --queues RESULT will print out the result queue only, while --q RESULT,LONGTHROW_DEPTH will print out all messages on two queues.  Do not put spaces around the commas, and they must be all caps.')
parser.add_argument('-s', '--strict', action='store_true', default=False,
                    help = 'Generate warnings if a message has a field which is neither optional nor required (ie. extra fields).')

parser.parse_args()
args = parser.parse_args()    

logging.basicConfig(datefmt='%Y-%m-%d %I:%M:%S',
                    format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG)

if args.queues is None:
    queue_names = None
else:
    queue_names = args.queues.split(',')
    
if args.messages is None:
    message_names = None
else:
    message_names = args.messages.split(',')
    
if args.containers is None:
    container_names = None
else:
    container_names = args.containers.split(',')     
    
containers = {}

for trippleStr in args.infile:
    # We ignore lines that start with # or are blank:
    if trippleStr[0] != '#' and len(trippleStr.strip()) > 0:
        #print(trippleStr)
        tripple = None
        try:
            tripple = json.loads(trippleStr)
            # Each line has three fields: queue, time_seconds, and message
        except Exception as e:
            print('')
            print(e)
            print(f'Error reading {trippleStr} from {tripple}')
            continue

        if 'queue' not in tripple:
            logging.error('Bad jsonl format.  No queue in the tripple.')
            continue
        if 'message' not in tripple:
            logging.error('Bad jsonl format.  No message in the tripple.')
            continue

        queue_name = tripple["queue"]
        message = tripple['message']                

        # if there is no type, then we skip it, but for other missing data, we continue.
        if 'type' not in message:
            logging.error('No type in this line so skipping: {trippleStr}')
            continue
        if 'uuid' not in message or message['uuid'] == '':
            logging.warning(f'No uuid in this line: {trippleStr}')
        if 'datetime' not in message or message['datetime'] == '':
            logging.warning(f'No datetime in this line: {trippleStr}')
        
        # First, filter the results
        if args.exclude:
            if queue_names is not None and queue_name in queue_names: 
                continue
            if message_names is not None and message['type'] in message_names:
                continue
            if 'container_name' in message and container_names is not None and message['container_name'] in container_names:
                continue                 
        else:
            if queue_names is not None and queue_name not in queue_names:
                continue
            if message_names is not None and message['type'] not in message_names:
                continue 
            if 'container_name' in message and container_names is not None and message['container_name'] not in container_names:
                continue 
            if 'container_name' not in message and container_names is not None:
                continue
        
        CCU.check_provinance(message, block_format=args.block_format, no_error_messages=args.no_error_messages) 
        if not args.provenance:    
            CCU.check_message(message, block_format=args.block_format, 
                              no_error_messages=args.no_error_messages, strict=args.strict)
        if not CCU.check_known(message['type']):
            if message['type'] in containers:
                 containers[message['type']] = containers[message['type']] + 1
            else:
                 containers[message['type']] = 1

if len(containers.keys()) > 0:                    
    sys.stdout.write('Unknown Messages by containers:')
    sys.stdout.write(json.dumps(containers, indent=4))

sys.exit(0)

