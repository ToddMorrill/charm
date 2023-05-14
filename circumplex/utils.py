import queue
import re
import time
import json
import os
import pickle

import tiktoken

# from googletrans import Translator

# speaker may be a multi-digit number or unknown
SPEAKER_RE = re.compile(r'Speaker \d+|Speaker unknown')
# SPEAKER_RE = re.compile(r'Speaker \d')  # extracts Speaker 1, Speaker 2, etc.
UTTERANCE_ID_RE = re.compile(
    r'\(\d+\)|\(\d+-\d+\)|\d+\.')  # extracts (1), (1-2), etc.
# extracts social orientation tags
SOCIAL_ORIENTATION_RE = re.compile(
    r'Assured-Dominant|Gregarious-Extraverted|Warm-Agreeable|Unassuming-Ingenuous|Unassured-Submissive|Aloof-Introverted|Cold|Arrogant-Calculating'
)
# extracts utterance id range, e.g. 20, 30 from (20-30)
UTTERANCE_ID_RANGE_RE = re.compile(r'(\d+)')


class Label(object):
    """Takes a label string from GPT and extracts the speaker, utterance id
    range, and social orientation tag(s). must pass pre-compiled regexes."""

    def __init__(self,
                 file_id,
                 label_str,
                 speaker_re=SPEAKER_RE,
                 utterance_id_re=UTTERANCE_ID_RE,
                 social_orientation_re=SOCIAL_ORIENTATION_RE,
                 utterance_id_range_re=UTTERANCE_ID_RANGE_RE):
        self.file_id = file_id
        self.label_str = label_str
        self.speaker_re = speaker_re
        self.utterance_id_re = utterance_id_re
        self.social_orientation_re = social_orientation_re
        self.utterance_id_range_re = utterance_id_range_re

        self.speaker_id = None
        self.utterance_id = None
        self.social_orientation = None
        self.speaker_id_count = 0
        self.utterance_id_count = 0
        self.social_orientation_count = 0
        self.utterance_id_start = None
        self.utterance_id_end = None
        self.is_range = False
        self.data = []

        self._parse()
        self._clean()
        self._get_data()

    def _parse(self):
        self._speaker_id = list(self.speaker_re.finditer(self.label_str))
        self._utterance_id = list(self.utterance_id_re.finditer(
            self.label_str))
        self._social_orientation = list(
            self.social_orientation_re.finditer(self.label_str))
        # default to first element, but might be useful to analyze multiple results later
        if len(self._speaker_id) > 0:
            self.speaker_id = self._speaker_id[0].group(0)
            self.speaker_id_count = len(self._speaker_id)
        if len(self._utterance_id) > 0:
            self.utterance_id = self._utterance_id[0].group(0)
            self.utterance_id_count = len(self._utterance_id)
        if len(self._social_orientation) > 0:
            self.social_orientation = self._social_orientation[0].group(0)
            self.social_orientation_count = len(self._social_orientation)

    def __repr__(self):
        return f'Label({self.file_id}, {self.label_str})'

    def _clean(self):
        # extract speaker number if present
        if self.speaker_id is not None:
            self.speaker_id = self.speaker_id.split(' ')[-1]
            if self.speaker_id != 'unknown':
                self.speaker_id = int(self.speaker_id)
        # extract utterance id range if present
        if self.utterance_id is not None:
            utterance_ids = list(
                self.utterance_id_range_re.finditer(self.utterance_id))
            if len(utterance_ids) > 0:
                self.utterance_id_start = int(utterance_ids[0].group(0))
            if len(utterance_ids) > 1:  # i.e. a range
                self.utterance_id_end = int(utterance_ids[1].group(0))
                self.is_range = True

    def _get_data(self):
        if self.is_range:
            for i in range(self.utterance_id_start, self.utterance_id_end + 1):
                self.data.append((self.file_id, self.social_orientation, i, self.speaker_id, self.label_str))
        else:
            self.data.append((self.file_id, self.social_orientation, self.utterance_id_start, self.speaker_id, self.label_str))

# class GoogleTranslator(object):
#     def __init__(self, src='zh-cn', dest='en'):
#         self.translator = Translator()
#         self.src = src
#         self.dest = dest
        
#     def translate(self, input_texts):
#         if not isinstance(input_texts,list):
#             input_texts = [input_texts]
        
#         target_texts = []
#         for text in input_texts:
#             translation = self.translator.translate(text, src=self.src, dest=self.dest)
#             target_texts.append(translation.text)
#         return target_texts

def translate_batch(sentences):
    translator = Translator()

    completed = [None]*len(sentences)
    errors = [] # keep track of errors
    error_count = 0 # if error count exceeds 100, stop the program and inspect
    # get a sentence from the queue, translate, check if translation column has text in it or not
    durations = [] 
    for idx, sentence in enumerate(sentences): # iterate until we get the sentinel of None
        start = time.time()
        # print(f'Translating: {sentence}')
        try:
            translation = translator.translate(text=sentence, src='zh-cn', dest='en')
            completed[idx] = translation.text
        except Exception as e:
            error = (idx, sentence, e)
            print(error)
            errors.append(error)
            error_count += 1
            ## check on the program at this point, something might be majorly wrong
            if error_count > 1000:
                break
        end = time.time()
        duration = end - start
        durations.append(duration)
    lens = [len(s) for s in sentences]
    print(f"Time taken to translate {len(sentences) - len(errors)} utterances, with average length {sum(lens)/len(lens):.2f}: {duration:.2f} seconds")
    print(f'Time per translation: {sum(durations) / (len(sentences) - len(errors)): .2f}')
        
    # rerun the errors
    print(f'Errors: {len(errors)}')
    while len(errors) > 0:
        idx, sentence = errors.pop(0)
        try:
            translation = translator.translate(text=sentence, src='zh-cn', dest='en')
            completed[idx] = translation.text
        except Exception as e:
            error = (idx, sentence, e)
            print(error)
            errors.append(error)
            error_count += 1
    return completed

def set_api_key(credentials_filepath='openai_credentials.json', user='todd'):
    openai_creds = json.load(open(credentials_filepath, 'r'))
    creds = openai_creds[user]
    os.environ['OPENAI_API_KEY']=creds['OPENAI_API_KEY']
    os.environ['OPENAI_ORG_ID']=creds['OPENAI_ORG_ID']

def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        # print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        # print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens