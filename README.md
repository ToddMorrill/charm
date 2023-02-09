# charm
DARPA CHARM project repo

### Evaluation
Verify the correctness of the reference directory:
```
# TODO:
```

### Stripping LDC headers from .mp4 files
```
INPUT_FILE=~/Documents/data/charm/raw/LDC2022E22_CCU_TA1_Mandarin_Chinese_Mini_Evaluation_Source_Data/data/video/M01003YBL.mp4.ldcc
OUTPUT_FILE=M01003YBL.mp4
dd if=$INPUT_FILE of=$OUTPUT_FILE ibs=1024 skip=1

INPUT_FILE=~/Documents/data/charm/raw/LDC2022E22_CCU_TA1_Mandarin_Chinese_Mini_Evaluation_Source_Data/data/video/M010040RH.mp4.ldcc
OUTPUT_FILE=M010040RH.mp4
dd if=$INPUT_FILE of=$OUTPUT_FILE ibs=1024 skip=1
tail --bytes=+1025 $INPUT_FILE > $OUTPUT_FILE
```

### Whisper
```
whisper M010040RH.mp4 --language Chinese --task translate --model large
```
### Data
Text conversations can be downloaded from Google Drive [here](https://drive.google.com/drive/folders/1y3JdISN1EapNNxGM_mMN_xsQ2nuIgJiY) and then the XML conversation files can be prepared into text documents using
```
INPUTDIR=~/Documents/datasets/charm/raw/ltf
OUTPUTDIR=~/Documents/datasets/charm/transformed/rsd
<!-- mkdir -p $OUTPUTDIR -->
./ltf2rsd.perl -o $OUTPUTDIR $INPUTDIR
``` 

### Environment setup
You can install the necessary Python dependencies (in a new virtual environment) by running
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m ipykernel install --user --name charm --display-name "Python (CHARM)"

# optionally launch Jupyter Lab
jupyter lab
```