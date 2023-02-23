import json
import random
import torch

random.seed(36)
import os
from xml.dom import minidom
import uuid
import numpy as np


class TextData:
    def __init__(self, id, start, end, text, uuid):
        self.id = id
        self.start = start
        self.end = end
        self.text = text
        self.uuid = uuid

    def __str__(self):
        return 'id={}, start={}, end={}, text={} uuid={}'.format(
            self.id, self.start, self.end, self.text, self.uuid)


class CCUDataLoader:

    def __init__(self, data_dir, transcript_root=None):
        self.text_root = os.path.join(data_dir, 'text')
        self.video_root = os.path.join(data_dir, 'video')
        self.audio_root = os.path.join(data_dir, 'audio')
        self.transcript_root = transcript_root
        self.current_source = None

    def load_text(self, file_id):
        file_path = os.path.join(self.text_root, 'ltf', f'{file_id}.ltf.xml')
        self.current_source = file_path
        file = minidom.parse(file_path)
        models = file.getElementsByTagName('SEG')

        data = []
        for i, model in enumerate(models):
            dataObject = TextData(
                str(model.attributes['id'].value),
                int(model.attributes['start_char'].value),
                int(model.attributes['end_char'].nodeValue),
                str(model.childNodes[1].firstChild.nodeValue),
                str("text_{}_{}".format(file_id, i)))
            data.append(dataObject)

        return data

    def load_transcript(self, document_type, document_name):

        if document_type == "video":
            transcript_root = "{}video/".format(self.transcript_root)
            self.current_source = self.video_root + document_name + ".mp4.ldcc"
        elif document_type == "audio":
            transcript_root = "{}audio/".format(self.transcript_root)
            self.current_source = self.audio_root + document_name + ".flac.ldcc"
        else:
            raise ValueError

        input_source = "{}{}_processed_results.json".format(
            transcript_root, document_name)
        f = open(input_source)
        json_data = json.load(f)
        if 'asr_preprocessed_turn_lvl' in json_data:
            utterances = json_data['asr_preprocessed_turn_lvl']
        else:
            utterances = json_data['asr_turn_lvl']
        # print("doc:", document_type, document_name, "| len:", len(utterances))
        data = []
        for i, utterance in enumerate(utterances):
            dataObject = TextData(
                str("{}_{}_{}".format(document_type, document_name, i)),
                float(utterance['start_time']), float(utterance['end_time']),
                str(utterance['transcript']),
                str("{}_{}_{}".format(document_type, document_name, i)))
            data.append(dataObject)

        return data


def get_turn_text(message, debug=False):

    if 'asr_text' not in message:
        if debug:
            print("Warning: ASR message contains no text field")
        return None
    elif 'asr_type' not in message:
        if debug:
            print("Warning: ASR message contains no type field")
        return None
    elif message['asr_type'] != "CONSOLIDATED_RESULT":
        if debug:
            print("Wrong message type", message['asr_type'])
        return None

    turn_text = message['asr_text']

    if not type(turn_text) == str:
        if debug:
            print("Warning: Text field in ASR message has type - ",
                  type(turn_text))
        return None
    if "@reject@" in turn_text:
        turn_text = turn_text.replace("@reject@", "")
    turn_text = turn_text.rstrip().lstrip()
    if turn_text == None or turn_text == '':
        if debug:
            print("Warning: Text field in ASR message is empty or None")
        return None

    return turn_text


def get_labels(model_name):

    f = open("model_store/" + model_name + "/labels.json", )
    label_json = json.load(f)
    final_labels = dict()
    for k, v in label_json.items():
        final_labels[int(k)] = v
    return final_labels


def get_test_messages(test_item='test0'):

    filename = "data/" + str(test_item) + ".jsonl"

    print("Using test dataset", test_item)

    with open(filename, 'r') as json_file:
        json_list = list(json_file)

    messages = []
    for json_str in json_list:
        result = json.loads(json_str)
        messages.append(result)
    return messages


def get_test_ASR(doc_type="text", doc_name="M01000FLX"):

    filename = "new_data/{}_{}.jsonl".format(doc_type, doc_name)

    print("Loading from", filename)

    with open(filename, 'r') as json_file:
        json_list = list(json_file)

    messages = []
    for json_str in json_list:
        result = json.loads(json_str)
        messages.append(result)
    return messages


def generate_test_messages():
    message = {
        "type": "ASR",
        "uuid": "578ca98c-557e-4ba8-8583-1eeaf31e7c93",
        "datetime": "2022-04-07 22:58:12.583927",
        "text": "故意杀人罪 ， 枪毙 ！ [ 怒 ] / / @ 贵 sir : 转发 微博",
        "source": "Operator",
    }
    messages = [message for _ in range(20)]
    message = {
        "type": "ASR",
        "uuid": "578ca98c-557e-4ba8-8583-1eeaf31e7c93",
        "datetime": "2022-04-07 22:58:12.583927",
        "text":
        "好吃 P ， 无药医 ~   [ 偷笑 ] / / @ 坚果 俱乐部 _ 老鬼 : 你 怎么 能 让 猫 看到 鱼 呢 ?",
        "source": "Operator",
    }
    messages += [message for _ in range(20)]
    random.shuffle(messages)
    return messages


def load_emotion_model(device):

    emotion_model_name = "nlpcc2014"
    print("Loading emotion model trained from", emotion_model_name)
    emotion_model_root = "model_store/" + emotion_model_name + "/"
    emotion_model_dict_root = emotion_model_root + "model_dict.pt"
    emotion_tokenizer_root = emotion_model_root + "tokenizer.pt"

    emotion_labels = get_labels(emotion_model_name)
    emotion_model = torch.load(emotion_model_dict_root, map_location=device)
    emotion_model.to(device)
    emotion_tokenizer = torch.load(emotion_tokenizer_root)

    return emotion_model, emotion_tokenizer, emotion_labels


def load_failure_model(failure_model, device):

    failure_models = [
        "fine_tuned_roberta",
        "fine_tuned_mengzi_t5",
        "zero_shot_erlangshen",
        "entrainment_mandarin",
    ]
    assert failure_model in failure_models
    failure_model_id = failure_models.index(failure_model)
    failure_models_dir = [
        "/local/nlp/huangyk/model_save/roberta_zh_ft",
        "model_store/mandarin_failure/t5_zh_ft",
        None,
        "model_store/mandarin_failure/entrainment",
    ]

    # failure_model_name = failure_models[failure_model_id]
    failure_model_dir = failure_models_dir[failure_model_id]
    failure_model = get_mandarin_model(model_name=failure_model,
                                       device=device,
                                       model_dir=failure_model_dir)
    return failure_model


class DialogAct:

    def __init__(self, text, start, end, score=None):
        self.text = text
        self.start = start
        self.end = end
        self.score = score
        self.change_score = 0.0

    def __str__(self):
        return "start: {}, end: {}, score: {} \ntext: {}".format(
            self.start, self.end, self.score, self.text)


class ChangePoint:

    def __init__(self, llr, tone, turn):
        assert tone in ["negative", "positive"]
        self.llr = llr
        self.tone = tone
        self.turn = turn

    def __str__(self):
        return "llr: {}, tone: {}, \nturn:\n{}".format(self.llr, self.tone,
                                                       self.turn)


def check_running_avg(scores, block_size=16, diff=0.5):
    if len(scores) < 2 * block_size:
        return None
    comp_block = scores[-(block_size * 2):]
    assert len(comp_block) == (2 * block_size)
    left_block = comp_block[:block_size]
    right_block = comp_block[block_size:]
    assert len(left_block) == len(right_block) == block_size
    running_diff = np.mean(right_block) - np.mean(left_block)
    return running_diff


def check_last_change(found_changepoints):
    return found_changepoints[-1].tone


def calc_llr(scores, block_size, tone):

    comp_block = scores[-(block_size * 2):]
    assert len(comp_block) == (2 * block_size)
    left_block = comp_block[:block_size]
    right_block = comp_block[block_size:]
    llr = np.log(np.mean(left_block) / np.mean(right_block))
    return llr


def check_for_change(
    turns,
    found_changepoints,
    threshold=0.15,
    block_size=16,
    method="running_avg",
):
    new_change = False
    scores = [turn.score for turn in turns]
    if method == "running_avg":
        running_avg = check_running_avg(scores, block_size=block_size)
        if len(turns) >= block_size:
            if running_avg:
                turns[-block_size].change_score = running_avg
            else:
                turns[-block_size].change_score = 0
        if running_avg and running_avg > threshold:
            if len(found_changepoints) > 0:
                last_change = check_last_change(found_changepoints)
                if last_change == "positive":
                    llr = calc_llr(scores, block_size, "negative")
                    found_changepoints.append(
                        ChangePoint(llr, "negative", turns[-block_size]))
                    new_change = True
            else:
                llr = calc_llr(scores, block_size, "negative")
                found_changepoints.append(
                    ChangePoint(llr, "negative", turns[-block_size]))
                new_change = True

        if running_avg and running_avg < (-1 * threshold):
            if len(found_changepoints) > 0:
                last_change = check_last_change(found_changepoints)
                if last_change == "negative":
                    llr = calc_llr(scores, block_size, "positive")
                    found_changepoints.append(
                        ChangePoint(llr, "positive", turns[-block_size]))
                    new_change = True
            else:
                llr = calc_llr(scores, block_size, "positive")
                found_changepoints.append(
                    ChangePoint(llr, "positive", turns[-block_size]))
                new_change = True
    else:
        raise ValueError
    return new_change

def load_whisper_transcript():
    pass