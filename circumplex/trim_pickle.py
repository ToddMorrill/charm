"""Slim down the pickle file containing the change point data. Should have
gone with the smaller version when it was created.
"""
import os
import pickle
import utils
from tqdm import tqdm

def main():
    # load the pickle file
    data = utils.load_pickle(os.path.expanduser('~/Documents/data/charm/transformed/tm3229-cache.pkl'))
    for file_id in tqdm(data):
        if not data[file_id]['processed']:
            continue
        if data[file_id]['data_type'] == 'text':
            continue
        for utterance in data[file_id]['utterances']['whisper']:
            utterance.pop('audio_files')
            utterance.pop('video_frames')

    out_filepath = '~/Documents/data/charm/transformed/tm3229-cache-slim.pkl'
    with open(os.path.expanduser(out_filepath), 'wb') as f:
        pickle.dump(data, f)
    
if __name__ == '__main__':
    main()