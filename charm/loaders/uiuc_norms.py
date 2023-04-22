import os
import sys
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt


NORMS = {
    101: 'apology',
    102: 'criticism',
    103: 'greeting',
    104: 'request',
    105: 'persuasion',
    106: 'thanks',
    107: 'taking leave',
    108: 'humour',
    109: 'embarrassment',
    110: 'command or order',
    111: 'congrats',
    112: 'interest',
    113: 'concern',
    114: 'encouragement',
    115: 'empathy',
    116: 'feedback',
    117: 'trust',
    118: 'respect',
    119: 'flattery'
}


def load_uiuc_norms(norms_dir):
    norms = {}
    for tab_file in glob.glob(os.path.join(norms_dir, '*tab')):
        if 'system_output' in tab_file:
            continue

        file_id, file_norms = os.path.basename(tab_file).split('.')[0], []
        for _, row in pd.read_csv(tab_file, sep='\t').iterrows():
            assert row['status'] in {'adhere', 'violate'}
            file_norms.append({
                'start': float(row['start']),
                'end': float(row['end']),
                'status': row['status'],
                'norm': NORMS[int(row['norm'])],
                'llr': float(row['llr'])
            })

        assert file_id not in norms

        norms[file_id] = list(sorted(
            file_norms, key=lambda x: x['start']
        ))
    return norms


if __name__ == '__main__':
    norms_dir = sys.argv[1]
    norms = load_uiuc_norms(norms_dir)

    plt.hist([norm['llr'] for file_id in norms for norm in norms[file_id]], bins=10)
    plt.show()

    for file_id in norms:
        for norm in norms[file_id]:
            if norm['start'] >= norm['end']:
                print(file_id, norm)
                break

    test_ids, val_ids = set(), set()
    with open('loaders/test.json', 'r') as f:
        test_ids.update(json.load(f))

    with open('loaders/val.json', 'r') as f:
        val_ids.update(json.load(f))

    len_train = len(set(norms.keys()) - test_ids - val_ids)
    len_val = len(set(norms.keys()) & val_ids)
    len_test = len(set(norms.keys()) & test_ids)

    print('Train: %d, Val: %d, Test: %d' % (len_train, len_val, len_test))

