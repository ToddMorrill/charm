# charm
DARPA CHARM project repo

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