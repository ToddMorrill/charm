#!/bin/bash

# You should have started ccuhub.py, logger.py and main.py in separate terminal windows
python script.py --jsonl ./transcripts/audio_M01000537_input.jsonl --fast 0.01
python script.py --jsonl ./transcripts/text_M01000FLX_input.jsonl --fast 0.01
python script.py --jsonl ./transcripts/video_M010009A4_input.jsonl --fast 0.01
python script.py --jsonl ./transcripts/text_M01000FLY_input_translation.jsonl --fast 0.01
python script.py --jsonl ./transcripts/audio_M01000538_input_translation.jsonl --fast 0.01
python script.py --jsonl ./transcripts/video_M010009BC_input_translation.jsonl --fast 0.01
python script.py --jsonl ./transcripts/test_frontend_input.jsonl --fast 0.01

