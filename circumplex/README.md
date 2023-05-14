# Circumplex Theory
The circumplex theory work attempts to assign social orientation tags {Assured-Dominant, Gregarious-Extraverted, Warm-Agreeable, Unassuming-Ingenuous, Unassured-Submissive, Aloof-Introverted, Cold, Arrogant-Calculating} to utterances in a conversation with the hopes of:
1. Developing more explainable models for change point detection
2. Augmenting the change point data from LDC
3. Providing novel signal for an ensemble system

## GPT Data Annotation
`GPT_annotation.ipynb` walks through some preliminary assessments of different language models for the social orientation tagging task with a single sample conversation.

`patch_speakers.ipynb` pulls in speaker information for all text conversations, where available, and updates tm3229-cache.pkl.

`prepare_text_conversatons.ipynb` generates all the requests that will be sent to OpenAI's API for the social orientation tagging task.

`api_request_parallel_processor.py` is a script that sends requests to OpenAI's API in parallel.

`analyze_gpt_annos.ipynb` analyzes the results of the GPT annotation task.

`prompt.txt` contains the prompt used for interactions with the OpenAI model.
- would need to do some digging to determine which file_id the current example in the prompt is from
- it's definitely from a text conversation where we knew the speaker IDs

`utils.py` contains helpful supporting code such as the parser for the GPT response.

`prompt_speaker_unknown.txt` is an addendum you can use to get annotations for speaker IDs (and social orientation tags) for conversations where you don't know the speaker IDs
- this transcript is from LDC2022E11_CCU_TA1_Mandarin_Chinese_Development_Source_Data_R1/data/audio/M0100053I.flac.ldcc
- there is a change point occurring in the middle of this transcript

