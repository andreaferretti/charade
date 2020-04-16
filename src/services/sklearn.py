import os, sys
import pickle
import json

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from services import Service, MissingLanguage, MissingParameter, MissingModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from common.sklearn.classification.features import split_text, get_features


class Classifier(Service):
    def __init__(self, model_dir='models/sklearn/classification'):
        Service.__init__(self, 'classification', 'sklearn', [])

        self.models = {}
        self.patterns = {}
        self.extra_patterns = {}
        self.results = {}
        langs = set()
        for name in os.listdir(model_dir):
            model_path = os.path.join(model_dir, name, 'model.pkl')
            results_path = os.path.join(model_dir, name, 'results.json')
            with open(model_path, 'rb') as f:
                m_ = pickle.load(f)
                self.models[name] = m_['model']
                self.patterns[name] = m_['patterns']
                self.extra_patterns[name] = m_['extra_patterns']
            with open(results_path) as f:
                results = json.load(f)
                if 'lang' in results:
                    langs.add(results['lang'])
                self.results[name] = results
        self.langs = list(langs)

    def run(self, request, response):
        text = request['text']
        lang = request['lang']
        debug = request.get('debug', False)
        if lang not in self.langs:
            raise MissingLanguage(lang)

        if 'classification-model' in request:
            model_name = request['classification-model']
        else:
            raise MissingParameter(self.task, self.name, 'classification-model')

        if model_name in self.models:
            model = self.models[model_name]
            patterns = self.patterns[model_name]
            extra_patterns = self.extra_patterns[model_name]
        else:
            raise MissingModel(self.task, self.name, model_name, list(self.models.keys()))

        features = np.array([get_features(text.lower(), patterns, extra_patterns)])
        category = model.predict(features)[0]
        probs = model.predict_proba(features)[0]
        category_prob = probs.max()
        result = {
            'category': category,
            'category_probability': category_prob
        }
        if debug:
            result['distribution'] = dict(zip(model.classes_, probs))
        return result

    def describe(self):
        result = super().describe()
        result['langs'] = self.langs
        result['models'] = self.results
        result['extra-params'] = [{
            'name': 'classification-model',
            'type': 'string',
            'choices': list(self.models.keys()),
            'required': True
        }]
        return result


class NMF(Service):
    def __init__(self, model_dir=os.path.join('models', 'sklearn', 'nmf')):
        Service.__init__(self, 'topic-modeling', 'sklearn', [])
        self.models = {}
        self.vectorizers = {}
        self.results = {}
        langs = set()
        for name in os.listdir(model_dir):
            model_path = os.path.join(model_dir, name, 'model.pkl')
            results_path = os.path.join(model_dir, name, 'results.json')
            with open(model_path, 'rb') as f:
                m_ = pickle.load(f)
                self.models[name] = m_['model']
                self.vectorizers[name] = m_['vectorizer']
            with open(results_path) as f:
                results = json.load(f)
                if 'lang' in results:
                    langs.add(results['lang'])
                self.results[name] = results
        self.langs = list(langs)

    def run(self, request, response):
        text = request['text']
        lang = request['lang']
        debug = request.get('debug', False)
        if lang not in self.langs:
            raise MissingLanguage(lang)

        if 'nmf-model' in request:
            model_name = request['nmf-model']
        else:
            raise MissingParameter(self.task, self.name, 'nmf-model')

        if model_name in self.models:
            model = self.models[model_name]
            vectorizer = self.vectorizers[model_name]
        else:
            raise MissingModel(self.task, self.name, model_name, list(self.models.keys()))

        vectors = vectorizer.transform([text])
        probs = model.transform(vectors)
        category = int(probs.argmax())
        category_prob = probs.max()
        result = {
            'distribution': probs.flatten().tolist(),
            'best-topic': category,
            'best-score': category_prob
        }
        if debug:
            H1 = model.components_
            vocab = vectorizer.get_feature_names()
            result['topics'] = [[vocab[i] for i in np.argsort(x)[:-11:-1]] for x in H1]
        return result

    def describe(self):
        result = super().describe()
        result['langs'] = self.langs
        result['models'] = self.results
        result['extra-params'] = [{
            'name': 'nmf-model',
            'type': 'string',
            'choices': list(self.models.keys()),
            'required': True
        }]
        return result