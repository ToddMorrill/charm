import numpy as np


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


def get_turn_text(message, debug=False):
    """Validates and returns incoming message if it passes checks."""
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