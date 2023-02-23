#! /usr/bin/env python
# This program conects each queue's PUB process to all of the SUB processes.
# This is required, because otherwise PUB only goes to one SUB, and we need it to
# go to all of them.
# Parts of the code were inspired by this code (and by related discussion on the web):
# https://gist.github.com/kianby/e1d455e5fb2a14f8dee3c02c337527f5

import argparse
import logging
import threading

import zmq

from ccu import CCU

def proxy(xsub, xpub):

    global context

    frontend = context.socket(zmq.XPUB)
    frontend.bind(f"tcp://*:{xpub}")

    backend = context.socket(zmq.XSUB)
    backend.bind(f"tcp://*:{xsub}")

    zmq.proxy(frontend, backend)

    # proxy doesn't return, so not sure why the rest is needed.
    frontend.close()
    backend.close()
    context.term()

if __name__ == "__main__":
    
    logging.basicConfig(datefmt='%Y-%m-%d %I:%M:%S',
                    format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG)

    parser = argparse.ArgumentParser('Program to forward messages to all processes and containers listening for them.')
    parser.add_argument('-i', '--info', default=False, action='store_true',
                    # Python 3.9: action=argparse.BooleanOptionalAction, 
                    help = 'Print the ports which are being forwarded.')

    parser.parse_args()
    args = parser.parse_args()
    
    CCU.config()
    logging.info(f'CCU Hub running for version {CCU.__version__} on {CCU.host_ip}.')
    logging.info(f'with config file {CCU.config_file_path}.')
    
    context = zmq.Context()
    threads = []
    
    for name, queue in CCU.queues.items():
        if args.info:
            logging.info(f'for {name} forwarding XSUB {queue["PUB"]} to XPUB {queue["SUB"]}')
        threads.append(threading.Thread(target=proxy, args=(queue['PUB'], queue['SUB'])))
        
    for thread in threads:
        thread.start()