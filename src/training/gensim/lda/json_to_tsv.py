import json
import csv
import os
import sys

def translate(dataset, basedir=os.path.join('data', 'lda')):
    with open(os.path.join(basedir, f'{dataset}.json')) as f:
        data = json.load(f)

    max_key = int(list(data['content'].keys())[-1])

    with open(os.path.join(basedir, f'{dataset}.csv'), 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        for i in range(max_key):
            key = str(i)
            writer.writerow([data['content'][key], data['target'][key], data['target_names'][key]])

if __name__ == '__main__':
    translate(dataset=sys.argv[1], basedir=sys.argv[2])