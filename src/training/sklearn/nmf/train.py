import os
import csv, pickle
import json, datetime, time
import argparse

import numpy as np
from sklearn import decomposition
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def parse_options():
    parser = argparse.ArgumentParser(description='Run NMF model')
    parser.add_argument('--data', required=True, help='the TSV file that contains the training sentences and the labels')
    parser.add_argument('--num-topics', type=int, default=10, help='number of topics')
    parser.add_argument('--model-name', required=True, help='the name of the model to generate')
    parser.add_argument('--model-dir', default=os.path.join('models', 'sklearn', 'nmf'), help='the directory where to store the models')
    parser.add_argument('--stopwords-dir', default=os.path.join('resources', 'stopwords'), help='the directory with the stopwords')
    parser.add_argument('--lang', help='the language of the training sentences')
    return parser.parse_args()

if __name__ == '__main__':
    options = parse_options()

    training_date = datetime.datetime.now()
    training_start = time.monotonic()

    model_dir = os.path.join(options.model_dir, options.model_name)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'model.pkl')
    results_path = os.path.join(model_dir, 'results.json')

    with open(options.data) as f:
        reader = csv.reader(f, delimiter='\t')
        data = [line[0].lower() for line in reader]

    with open(os.path.join(options.stopwords_dir, f'{options.lang}.txt')) as f:
        stopwords = [line.strip() for line in f.readlines()]

    # Extract features
    print('Performing vectorization')
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(data)

    # Train model
    print('Training model')
    model = decomposition.NMF(n_components=options.num_topics)
    model.fit(vectors)

    training_elapsed = time.monotonic() - training_start

    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'vectorizer': vectorizer
        }, f)
    print(f'Model saved at {model_path}')

    # Save results
    with open(results_path, 'w') as f:
        results = {
            'lang': options.lang,
            'trained-at': training_date.isoformat(timespec='seconds'),
            'training-time': str(datetime.timedelta(seconds=training_elapsed)),
            'datasets': [options.data],
            'params': {
                'num_topics': options.num_topics,
                'vectorizer': 'TF-IDF',
                'model': 'NMF'
            },
            'metrics': {
                'reconstruction_error': model.reconstruction_err_
            }
        }
        results['params'].update(model.get_params())
        json.dump(results, f)
    print(f'Results saved as {results_path}')