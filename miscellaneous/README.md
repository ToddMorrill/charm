# Data Labeling

## Change Point Annotation Exercise (1/22/2023)
Annotation results [Google sheet](https://docs.google.com/spreadsheets/d/1RXrXGZ7TR4BcGzSpZZ5HvlPohHM4nm-CC8GiWMyI7Co/edit#gid=515959849)

**URL list:**
- https://www.bilibili.com/video/BV1Jv4y1c7Dv
- https://www.bilibili.com/video/BV1G4411H7Cc
- https://www.bilibili.com/video/BV1Uu41117BK

**Workflow:**
- Download file using this service: https://pastedownload.com/bilibili-video-downloader/ 
    - NB: had a hard time downloading an MP4 containing both audio and video so I just downloaded the audio
    - Another option is to download the raw source data in LDC2022E22 and unpack the .mp4.ldcc formatted data, though the corpus is ~120gb
- Transcribe (and optionally translate) the audio data using OpenAI's Whisper model
    - `pip install -U openai-whisper`
```
DATA_DIR=~/Documents/data/charm/transformed/annotations
whisper ${DATA_DIR}/1.mp4 ${DATA_DIR}/2.mp4 ${DATA_DIR}/3.mp4  --model large --language Chinese --output_dir ${DATA_DIR}
```
- Watch and listen to the video while looking at the transcription and/or translation
