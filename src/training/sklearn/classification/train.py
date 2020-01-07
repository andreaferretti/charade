# Custom model to classify documents.
# Works better with long text and few categories

import os, sys
import csv, pickle, random
import argparse
from collections import defaultdict, Counter
from itertools import product
from sklearn import linear_model
import numpy as np
import json, datetime, time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from common.sklearn.classification.features import split_text, get_features

def _freqs(counter, min_occurrencies=5, min_len=4):
    tokens = {k: v for k, v in counter.items() if v >= min_occurrencies and len(k) >= min_len}
    tokens_tot = sum(tokens.values())
    return {k: v/tokens_tot for k, v in tokens.items()}

def _ratio(m1, m2):
    res = dict()
    for k, v1 in m1.items():
        v2 = m2.get(k, 0)
        if v2 > 0:
            res[k] = v1 / v2
    return res

def _display_matrix(labels, predictions, classes):
    rows = []
    _count = Counter(zip(labels, predictions))
    _left_col_w = max(len(x) for x in classes) + 2
    row_format = ('{:<%s}' % _left_col_w) + len(classes)*'{:<5}'
    rows = [row_format.format('', *classes)]
    for c1 in classes:
        row = [c1] + [_count[(c1, c2)] for c2 in classes]
        rows.append(row_format.format(*row))
    return '\n'.join(rows)

def parse_options():
    parser = argparse.ArgumentParser(description='Run Custom classify model')
    parser.add_argument('--data', required=True, help='the TSV file that contains the training sentences and the labels')
    parser.add_argument('--extra-patterns', help='extra regexes to help the classifier')
    parser.add_argument('--bootstrap', type=int, default=200, help='how many samples of each category')
    parser.add_argument('--freq-threshold', type=int, default=20, help='min frequency ratio for each category pair')
    parser.add_argument('--topk', type=int, default=15, help='max number of tokens for each category pair')
    parser.add_argument('--column', type=int, default=1, help='column which contains the category')
    parser.add_argument('--test', default=False, help='add flag to perform a test after training', action='store_true')
    parser.add_argument('--model-name', required=True, help='the name of the model to generate')
    parser.add_argument('--model-dir', default=os.path.join('models', 'sklearn', 'classifier'), help='the directory where to store the models')
    parser.add_argument('--language', help='the language of the training sentences')
    return parser.parse_args()

if __name__ == '__main__':
    options = parse_options()

    training_date = datetime.datetime.now()
    training_start = time.monotonic()

    model_dir = os.path.join(options.model_dir, options.model_name)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'model.pkl')
    results_path = os.path.join(model_dir, 'results.json')

    extra_patterns = []
    if options.extra_patterns is not None and os.path.exists(options.extra_patterns):
        with open(options.extra_patterns) as f:
            extra_patterns = set([line.strip() for line in f.readlines()])

    with open(options.data) as f:
        reader = csv.reader(f, delimiter='\t')
        # first column is text, lowercase, second column is label
        data = [(line[0].lower(), line[options.column]) for line in reader]

    data_by_category = defaultdict(list)
    for text, c in data:
        data_by_category[c].append(text)
    classes = set(data_by_category)

    # Add extra patterns
    all_tokens_by_classes = {c: Counter() for c in data_by_category}
    for i in range(options.bootstrap):
        for cat, samples in data_by_category.items():
            text = random.choice(samples)
            tokens = split_text(text)
            all_tokens_by_classes[cat].update(tokens)

    all_freqs_by_category = {c: _freqs(ws) for c, ws in all_tokens_by_classes.items()}
    # for each pair of classes, find the most distinguishing token
    print('Learning useful patterns to distinguish between categories (topk: {topk}, threshold: {freq_threshold}, bootstrap: {bootstrap})'.format(
        topk=options.topk,
        freq_threshold=options.freq_threshold,
        bootstrap=options.bootstrap))
    test = defaultdict(list)
    threshold = options.freq_threshold
    for c1, c2 in product(classes, classes):
        if c1 != c2:
            frequencies_c1 = all_freqs_by_category[c1]
            frequencies_c2 = all_freqs_by_category[c2]
            ratios = sorted(_ratio(frequencies_c1, frequencies_c2).items(), key=lambda x: -x[1])
            for token, f in ratios:
                if f < threshold:
                    break
                test[(c1, c2)].append((token, f))
            test[(c1, c2)].sort(key=lambda x: -x[1])

    patterns = set()
    topk = options.topk
    for _, vs in test.items():
        patterns.update(v[0] for v in vs[:topk])
    patterns = sorted(patterns)

    # Extract features
    print('Extracting features with %s patterns and %s extra patterns' % (len(patterns), len(extra_patterns)))
    features_, labels = [], []
    for text, c in data:
        features_.append(get_features(text, patterns, extra_patterns))
        labels.append(c)
    features = np.array(features_)

    # Train model
    print('Training model')
    model = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg')

    # Compute accuracy
    if options.test:
        print('Testing model')
        model.fit(features, labels)
        predictions = model.predict(features)
        accuracy = (predictions == labels).mean()
        print('Accuracy: %.2f%%' % (100 * accuracy))

        # Compute confusion matrix
        print(_display_matrix(predictions, labels, sorted(classes)))

    training_elapsed = time.monotonic() - training_start

    # Save model
    model = {
        'model': model,
        'patterns': patterns,
        'extra_patterns': extra_patterns
    }

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print('Model saved at %s' % model_path)

    # Save results
    with open(results_path, 'w') as f:
        results = {
            'lang': options.language,
            'trained-at': training_date.isoformat(timespec='seconds'),
            'training-time': str(datetime.timedelta(seconds=training_elapsed)),
            'datasets': [options.data],
            'params': {
                'topk': options.topk,
                'threshold': options.freq_threshold,
                'bootstrap': options.bootstrap,
                'model': 'LogisticRegression',
                'solver': 'newton-cg'
            },
            'metrics': {
                'accuracy': accuracy
            }
        }
        json.dump(results, f)
    print(f'Results saved as {results_path}')
