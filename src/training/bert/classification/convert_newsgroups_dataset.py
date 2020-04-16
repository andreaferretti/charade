import pandas as pd
import os
import sys
import json


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
import common.bert.constants as constants


def convert(dataset, basedir=os.path.join('data', 'classification')):

    path_data = os.path.join(basedir, dataset)

    with open(os.path.join(path_data, f'{dataset}.json')) as f:
        data = json.load(f)

    max_key = int(list(data['content'].keys())[-1])

    rows = []
    for i in range(max_key):
        key = str(i)
        rows.append([data['content'][key], data['target'][key], data['target_names'][key]])

    index2category = dict(set([(category, category_name) for _, category, category_name in rows]))

    with open(os.path.join(path_data, constants.LABELS_FILE), 'w') as outfile:
        json.dump(index2category, outfile)

    data = pd.DataFrame(rows, columns=['sentence', 'category', 'category_name'])
    data[['sentence', 'category']].to_csv(os.path.join(path_data, 'train.csv'))

if __name__ == '__main__':
    convert(dataset=sys.argv[1], basedir=sys.argv[2])