# This script simulates events coming out of the Hololens.
# It simulates several operator move events and one operator pointing event.
# There is commented out code to simulate a frame from a video feed.
# (But this is not used because it is so large it makes it hard to follow the logs.)

import argparse
import json
import logging
import sys
import time

import zmq

from ccu import CCU


def open_pub_queues():
    for name, queue in CCU.queues.items():
        queue['socket'] = CCU.socket(queue, zmq.PUB) 

current_time = 0.0
    
parser = argparse.ArgumentParser('Program to send many messages for testing.')
parser.add_argument('-d', '--debug', default=False, action='store_true',
                    # Python 3.9: action=argparse.BooleanOptionalAction, 
                    help = 'Print out more details of each message sent.')
parser.add_argument('-f', '--fast', type=float, 
                    help = 'Send a message every argument seconds, ignoring when the message was actually recorded.  This argument can not be used with -s/-e arguments.')
parser.add_argument('infile', type=argparse.FileType('r', encoding='utf-8'), nargs='?', default=sys.stdin,
                    help = 'The file to use.  Stdin is used if no file is given.')
parser.add_argument('-j', '--jsonl', type=argparse.FileType('r', encoding='utf-8'),
                    help = 'Run the script in the provided file (in JSON line format).')
parser.add_argument('-q', '--quiet', default=False, action='store_true',
                    # Python 3.9: action=argparse.BooleanOptionalAction, 
                    help = 'Dont print out dots.')
parser.add_argument('-o', '--one', type=open, 
                    help = 'Inject the message in the provided file (in JSON format).')
parser.add_argument('-s', '--start_seconds', type=float, default=0.0,
                    help = 'Start time to send messsages (default=0.0).')
parser.add_argument('-e', '--end_seconds', type=float, default=-1,
                    help = 'Start time to send messsages (default=-1).')
parser.add_argument('-w', '--wiz', default=False, action='store_true',
                    # Python 3.9: action=argparse.BooleanOptionalAction, 
                    help = 'Send messages as fast as you can.  No sleeping between messages.  This argument can not be used with -s/-e arguments.')
# These arguments are used for passing messages one by one, which is a future feature.
#parser.add_argument('-M', '--message', type=str, 
#                    help = 'Inject the message in the provided file (in JSON format).')
#parser.add_argument('-Q', '--queue', type=str, 
#                    help = 'Inject the message in the provided file (in JSON format).')
#parser.add_argument('-W', '--wait', type=float, default=0.0, 
#                    help = 'Inject the message in the provided file (in JSON format).')

parser.parse_args()
args = parser.parse_args()    

CCU.config()
last_queue = None
last_wait = None
start_seconds = args.start_seconds
end_seconds = args.end_seconds

logging.basicConfig(datefmt='%Y-%m-%d %I:%M:%S',
                    format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG)

if args.one is not None:
    logging.info('Injecting message in {args.one} file.')
    message = json.load(args.one)
    print(message)
    #CCU.send(arg.queue, message)

# Users can either put the file name on the command line or with the -j option.
if args.jsonl is not None:
    jsonl_file = args.jsonl
else:
    jsonl_file = args.infile
    
if jsonl_file is not None:
    logging.info(f'Running script in {jsonl_file.name} file with CCU {CCU.__version__} on {CCU.host_ip}.')
    logging.info(f'with config file {CCU.config_file_path}.')  
    logging.info(f'Starting with messages at time {start_seconds}.')  
    if end_seconds > 0:
        logging.info(f'Ending with messages at time {end_seconds}.')  
    for trippleStr in jsonl_file:
        # We ignore lines that start with # or are blank:
        if trippleStr[0] != '#' and len(trippleStr.strip()) > 0:
            #print(trippleStr)
            tripple = None
            try:
                tripple = json.loads(trippleStr)
                # Each line has three fields: queue, time_seconds, and message
                queue_name = tripple["queue"]
                queue = CCU.queues[queue_name]
                trigger_time = tripple['time_seconds']
                message = tripple['message']                
            except Exception as e:
                print('')
                print(e)
                print(f'Error reading {trippleStr} from {tripple}')
                continue

            if trigger_time < start_seconds:
                if args.debug:
                    print(f'At {current_time} skipping {message["type"]} message on {queue_name}.')
                current_time = trigger_time
                continue

            if end_seconds > 0 and trigger_time > end_seconds:
                if args.debug:
                    print(f'At {current_time} stopping processing messages on {queue_name}.')
                break

            # We open sockets as we need them, so we don't waste time opening
            # sockets we do not need.
            if 'socket' not in queue:
                queue['socket'] = CCU.socket(queue, zmq.PUB)
            
            if not args.wiz and args.fast is None:
                if trigger_time > current_time:
                    time.sleep(trigger_time-current_time)
            elif args.fast is not None:
                time.sleep(args.fast)
            current_time = trigger_time
            
            # If messages don't have UUIDs or timestamps, then add them
            # JCL I'm not sure this is a good idea.
            if 'uuid' not in message or message['uuid'] == '':
                message['uuid'] = CCU.uuid()
                print('Warning: message in jsonl file does have a uuid field.')
            if 'datetime' not in message or message['datetime'] == '':
                message['datetime'] = CCU.now()             
                print('Warning: message in jsonl file does have a datetime field.')
            if args.debug:
                print(f'At {current_time} sending {message["type"]} message on {queue_name}.')
            CCU.send(queue['socket'], message)
            if not args.debug and not args.quiet:
                print('.', end='', flush=True)
    if not args.debug and not args.quiet:
        print('')
    
#if args.queue is not None and args.message is None:
#    print('No -M or --message to go with the -Q and -M/-Q must be provided together.')
#    sys.exit(1)
    
#if args.message is not None:
#    if args.queue is None:
#        print('No -Q or --queue option on command line and -M/-Q must be provided together.')
#        sys.exit(1)
#    queue = named_queues[args.queue]
#    time.sleep(args.wait)
#    current_time = current_time + args.wait
#    message_json = json.loads(args.message)
#    CCU.send(queue, message_json)
#    if not args.quiet:
#        print('.', end='')    
