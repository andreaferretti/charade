# From
# https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/

import os

from gensim.utils import simple_preprocess
from gensim.models.ldamodel import LdaModel

from services import Service, MissingModel, MissingParameter, MissingResource


def _preprocess(parsing, text, stopwords):
    tokens = []
    for sentence in parsing:
        for item in sentence:
            if 'text' in item:
                token = item['text']
            else:
                token = text[item['start']:item['end']]
            if any(c.isalpha() for c in token) and not token in stopwords:
                tokens.append(token)
    return [tokens]

class Lda(Service):
    def __init__(self, model_dir=os.path.join('models', 'gensim', 'lda'), stopwords_dir=os.path.join('resources', 'stopwords')):
        Service.__init__(self, 'topic-modeling', 'lda-gensim', ['parse'])
        self.models = {}
        self.stopwords = {}
        for name in os.listdir(model_dir):
            self.models[name] = LdaModel.load(os.path.join(model_dir, name, 'model'))
        for name in os.listdir(stopwords_dir):
            lang = name[:2]
            with open(os.path.join(stopwords_dir, name)) as f:
                self.stopwords[lang] = set([line.strip() for line in f.readlines()])

    def run(self, request, response):
        if 'lda-model' in request:
            model_name = request['lda-model']
        else:
            raise MissingParameter(self.task, self.name, 'lda-model')

        if model_name in self.models:
            model = self.models[model_name]
        else:
            raise MissingModel(self.task, self.name, model_name, list(self.models))

        lang = request['lang']
        if lang in self.stopwords:
            stopwords = self.stopwords[lang]
        else:
            raise MissingResource(self.task, self.name, f'{lang}.txt')

        text = request['text']
        parsing = response['parse']
        debug = request.get('debug', False)

        sentences = _preprocess(parsing, text, stopwords)
        corpus = [model.id2word.doc2bow(s) for s in sentences]
        distribution, _ = model.inference(corpus)
        distribution = distribution[0].astype('float')
        distribution /= distribution.sum()
        best_topic = distribution.argmax()
        best_score = distribution[best_topic]

        result = {
            'distribution': distribution.tolist(),
            'best-topic': int(best_topic),
            'best-score': float(best_score)
        }
        if debug:
            topics = []
            for i in range(model.num_topics):
                topic = [a for a, b in model.show_topic(i)]
                topics.append(topic)
            result['topics'] = topics

        return result

    def describe(self):
        result = super().describe()
        result['langs'] = list(self.stopwords.keys())
        result['extra-params'] = [{
            'name': 'lda-model',
            'type': 'string',
            'choices': list(self.models.keys()),
            'required': True
        }]
        result['models'] = {}
        for model_name, model in self.models.items():
            result['models'][model_name] = {
                'pretrained': False,
                'datasets': [f'{model_name}.csv']
            }
        return result