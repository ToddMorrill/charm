#!/usr/bin/env python
# This script checks CCU YAML config files

import glob
import sys
import yaml

known_behaviors = ['Slow-Start', 'Not-Realtime']
known_languages = ['en', 'zh']
known_resources = ['CPU', 'GPU-CPU', 'GPU', 'Internet']

fatal_errors = 0

def print_error(err_msg):
    global fatal_errors
    fatal_errors = fatal_errors + 1
    print(err_msg)

if sys.argv[1] != '':
    yaml_filenames = [ sys.argv[1] ]
else:
    yaml_filenames = glob.glob('*.yaml')

if sys.argv[1] == '--help':    
    print('python yamlcheck.py <ccu-yaml-file>')
    print('Checks a CCU YAML file to make sure it has required fields, and optional fields have the right data format.')
    sys.exit(0)
    
if len(yaml_filenames) == 0:
    print("No YAML files here.")
    sys.exit(1)
elif len(yaml_filenames) > 1:
    print("More than one YAML file.")
    sys.exit(1)
    
with open(yaml_filenames[0], 'r') as yaml_file:               
    yaml_data = yaml.safe_load(yaml_file)
    
# For debugging    
#print(yaml.dump(yaml_data, indent=4))

if 'name' not in yaml_data:
    print_error(f'ERROR: no name in YAML file and one is required.')
    
if 'version' not in yaml_data:
    print_error(f'ERROR: no version in YAML file and one is required.')  
    
if 'poc' not in yaml_data:
    print_error(f'ERROR: no poc (point of contact) in YAML file and one is required.')
else:
    if 'name' not in yaml_data['poc']:
        print_error(f'ERROR: no name in the poc (point of contact) in YAML file and one is required.')
    if 'email' not in yaml_data['poc']:
        print_error(f'ERROR: no email in the poc (point of contact) in YAML file and one is required.')
        
if 'testingInstructions' not in yaml_data:
    print(f'WARNING: no testingInstructions (testing instructions) in YAML file and some are required.')
    
if 'dockerCompose' not in yaml_data:
    print_error(f'WARNING: no dockerCompose section is in the YAML file it is required.')
    
if 'inputs' not in yaml_data:
    print_error(f'ERROR: no inputs in YAML file and they are required.')
    
if 'outputs' not in yaml_data:
    print_error(f'ERROR: no outputs in YAML file and they are required.')    

if 'name' in yaml_data and 'version' in yaml_data and 'poc' in yaml_data and \
   'name' in yaml_data['poc'] and 'email' in yaml_data['poc']:    
    print(f'Container {yaml_data["name"]}:{yaml_data["version"]} by {yaml_data["poc"]["name"]} ({yaml_data["poc"]["email"]})')

check_messages = []
if 'inputs' in yaml_data:
    queues = yaml_data['inputs']
    ###print(f'queues: {queues}')
    for queue in queues:
        ###print(f'queue: {queue}')
        #queue is a dictionary with one entry, so grab name and item:
        queue_name, queue_messages  = next(iter(queue.items()))
        ###print(f'In queue {queue_name} have messages: {queue_messages}')
        for message in queue_messages:
            ###print(f'message: {message}')
            check_messages.append((message,queue_name))
            ###print(f'  * CHECK: In messages and queues that {message} exists on the {queue_name} queue.')

if 'outputs' in yaml_data:
    queues = yaml_data['outputs']
    ###print(f'queues: {queues}')
    for queue in queues:
        ###print(f'queue: {queue}')
        #queue is a dictionary with one entry, so grab name and item:
        queue_name, queue_messages  = next(iter(queue.items())) 
        ###print(f'In queue {queue_name} have messages: {queue_messages}')
        for message in queue_messages:
            ###print(f'message: {message}')
            check_messages.append((message,queue_name))
            ###print(f'  * CHECK: In messages and queues that {message} exists on the {queue_name} queue.')        
    
if len(check_messages) > 0:
    print(f'Check the web page https://ldc-issues.jira.com/wiki/spaces/CCUC/pages/3060269057/Messages+and+Queues')
    print(f'that the listed messages are in the listed queues:')
    for check_message in check_messages:
        print(f'  * CHECK message {check_message[0]} is in queue {check_message[1]}.')

if 'resources' in yaml_data:
    resources = yaml_data['resources'] 
    if type(resources) is list:
        for resource in resources:
            if resource not in known_resources:
                print(f'WARNING: {resource} is not a known resource name.  {known_resources}') 
    else:            
        if resources not in known_resources:
            print(f'WARNING: {resources} is not a known resource name.  {known_resourcess}') 
            
if 'behaviors' in yaml_data:
    behaviors = yaml_data['behaviors']
    if type(behaviors) is list:
        for behavior in behaviors:
            if behavior not in known_behaviors:
                print(f'WARNING: {behavior} is not a known resource name.  {known_behaviors}')
    else:            
        if behaviors not in known_behaviors:
            print(f'WARNING: {behaviors} is not a known behavior name.  {known_behaviors}')                 
    
if 'trainingLanguage' in yaml_data:
    languages = yaml_data['trainingLanguage'] 
    if type(languages) is list:
        for language in languages:
            if language not in known_languages:
                print(f'WARNING: {language} is not a known language name.  {known_languages}')
    else:            
        if languages not in known_languages:
            print(f'WARNING: {languages} is not a known language name.  {known_languages}')   
            
sys.exit(fatal_errors)              