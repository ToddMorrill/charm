# Very simple example of a TA1 process. Not Realistic.
# This container listens for translations and emotions.
# If it gets an angry emotion with an LLR higher than 2.0, then all translations after
# that are considered rude, until another emotion occures or an angry with an LLR 2.0 or lower is seen.
# This example focuses on checking messages and adding the right trigger_id data.
import logging
import zmq
from ccu import CCU
logging.basicConfig(datefmt='%Y-%m-%d %I:%M:%S',

format='%(asctime)s %(levelname)s %(message)s',
level=logging.DEBUG)
logging.info('TA1 Example Running')
gesture = CCU.socket(CCU.queues['RESULT'], zmq.SUB)
result = CCU.socket(CCU.queues['RESULT'], zmq.PUB)
rude = False
emotion_uuid = None
while True:
# Waits for a translation.
# Start out by waiting for a message on the gesture queue.
message = CCU.recv_block(gesture)
# Once we get that message, see if it is a pointing message, since
# that is the only thing we care about.
if message['type'] == 'translation' or message['type'] == 'emotion':
# The first thing we should do is check that the message is legal:
CCU.check_message(message)
if message['type'] == 'emotion':
# This function test for the fields we are actually going to use, so if it fails
# we don't do anything else.
if not CCU.check_message_contains(message,['name', 'llr', 'uuid']):
logger.warning('Got an emotion message which did not have a name or a llr, so ignoring it.')
continue
if message['name'] == 'angry' and message['llr'] > 2.0:
rude = True
emotion_uuid = message['uuid']
else:

5

rude = False
emotion_uuid = None
if message['type'] == 'translation':
# This function test for the fields we are actually going to use, so if it fails
# we don't do anything else.
if not CCU.check_message_contains(message,['asr_type', 'asr_text', 'uuid']):
logger.warning('Got an asr_result message which did not have a asr_type or a asr_text, so ignoring it.')
continue
logging.debug(f'Found an asr_result with rudeness {rude}.')
if rude:
# Creates a simple messsage of type 'rudeness_detected'.
# The base_message function populates the 'uuid'
# and 'datetime' fields in the message (which is really just a
# Python dictionary).
new_message = CCU.base_message('rudeness_detected',
text=message['asr_text'],
provenance=message['uuid'],
trigger_id=[message['uuid'], emotion_uuid])

# Shows how to add fields to an existing message and also how to check for fields in
# a message one at a time.
if 'start_seconds' in message:
new_message['start_seconds'] = message['start_seconds']
else:
logger.warning('Translation message does not have a start_seconds field, which was expected.')
if 'end_seconds' in message:
new_message['end_seconds'] = message['end_seconds']
else:
logger.warning('Translation message does not have a end_seconds field, which was expected.')
# Then we check that it has the required fields, and does not have any extra fields.
# But even if it does, we still send it out. (Maybe we should not but we do.)
if not CCU.check_message(new_message, strict=False):
logger.warning('Created a message without the required fields, but sending it anyway.')
# Finally, we send it.
CCU.send(result, new_message)


# RESULT channel is useful for TA1 output
# pull container name from environment variable
# trigger_id field signifies the data that caused another message
# can we expect to have access to ASR Result ("asr_result") Message?