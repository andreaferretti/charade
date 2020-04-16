import pandas as pd
import os
import sys
import json


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
import common.bert.constants as constants


def get_sentiment(x):
    if x['opos'] == 0 and x['oneg'] == 1:
        sentiment = 0  # negative
    elif x['opos'] == 1 and x['oneg'] == 0:
        sentiment = 1  # positive
    elif x['opos'] == 1 and x['oneg'] == 1:
        sentiment = 2  # mixed
    else:
        sentiment = 3  # neutral

    return sentiment



def convert(dataset, basedir=os.path.join('data', 'classification')):

    path_data = os.path.join(basedir, dataset)
    data = pd.read_csv(path_data)

    data['sentiment'] = data.apply(lambda x: get_sentiment(x), axis=1)
    data['sentiment'].value_counts()
    data.to_csv(os.path.join(path_data))

    index2category = {0: 'negative', 1: 'positive', 2: 'mixed', 3: 'neutral'}
    with open(os.path.join(basedir, constants.LABELS_FILE), 'w') as outfile:
        json.dump(index2category, outfile)



if __name__ == '__main__':
    convert(dataset=sys.argv[1], basedir=sys.argv[2])