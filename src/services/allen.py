import os
import json

import torch
from allennlp.models import Model
from allennlp.predictors import Predictor
from allennlp.common.params import Params
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.fields.text_field import TextField
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper

from services import Service, MissingLanguage
from common.allen.ner.wikiner import WikinerDatasetReader
from common.allen.sentiment.model import LstmClassifier


def _entity_iterator(ner_prediction, text, offset = 0, default_tag='O'):
    words = ner_prediction['words']
    tags = ner_prediction['tags']
    assert(len(words) == len(tags))
    for i in range(len(words)):
        word = words[i]
        tag = tags[i]
        start = text.index(word, offset)
        end = start + len(word)
        offset = end
        yield start, end, tag

def _is_continuing_tag(prev_tag, tag):
    return (prev_tag[:2] == 'B-') and (tag[:2] == 'I-') and (prev_tag[2:] == tag[2:])

def _clean_tag(tag):
    if tag[:2] in ['I-', 'B-']:
        return tag[2:]
    else:
        return tag

def _to_annotations(ner_prediction, text, offset = 0, debug=False, default_tag='O'):
    response = []
    last_tag = default_tag
    last_start = 0
    last_end = 0
    for start, end, tag in _entity_iterator(ner_prediction, text, offset, default_tag):
        if tag != last_tag and not _is_continuing_tag(last_tag, tag):
            if last_tag != default_tag:
                item = {
                    'start': last_start,
                    'end': last_end,
                    'label': _clean_tag(last_tag)
                }
                if debug:
                    item['text'] = text[last_start: last_end]
                response.append(item)
            last_start = start
            last_end = end
        else:
            last_end = end
        last_tag = tag

    if last_tag != default_tag:
        item = {
            'start': last_start,
            'end': last_end,
            'label': _clean_tag(last_tag)
        }
        if debug:
            item['text'] = text[last_start: last_end]
        response.append(item)

    return response


def _load_model_description(model_dir):
    """
    Reads configuration parameters and training metrics from JSON files stored with the AllenNLP model.
    Returns a dictionary that can be used to populate the 'models' field of the service description.
    """
    desc = {'pretrained': False}
    try:
        config_fn = os.path.join(model_dir, "config.json")
        metrics_fn = os.path.join(model_dir, "metrics.json")
        if os.path.isfile(config_fn):
            with open(config_fn) as config_file:
                config = json.load(config_file)
                desc['datasets'] = [config.get('train_data_path', '')]
                desc['params'] = config.get('trainer')
        if os.path.isfile(metrics_fn):
            with open(metrics_fn) as metrics_file:
                metrics = json.load(metrics_file)
                desc['training-time'] = metrics.get('training_duration')
                desc['metrics'] = metrics
    except (IOError, ValueError):
        pass
    return desc

class PretrainedNer(Service):
    def __init__(self):
        Service.__init__(self, 'ner', 'allen', [])
        self.model = Predictor.from_path('models/allen/pretrained/ner-model-2020.02.10.tar.gz')

    def run(self, request, response):
        if request['lang'] == 'en':
            text = request['text']
            debug = request.get('debug', False)
            prediction = self.model.predict(sentence=text)
            return _to_annotations(prediction, text, debug=debug)
        else:
            raise MissingLanguage(request['lang'])

    def describe(self):
        result = super().describe()
        result['langs'] = ['en']
        result['models'] = {'en': {'pretrained': True}}
        return result

class Ner(Service):
    def __init__(self, models_dir='models/allen/ner'):
        Service.__init__(self, 'ner', 'allen-custom', ['parse'])
        self.readers = {}
        self.predictors = {}
        self.descriptions = {}
        for lang in os.listdir(models_dir):
            reader, predictor = self._load_reader_and_predictor(os.path.join(models_dir, lang))
            self.readers[lang] = reader
            self.predictors[lang] = predictor
            self.descriptions[lang] = _load_model_description(os.path.join(models_dir, lang))

    @staticmethod
    def _load_reader_and_predictor(basedir):
        config_path = os.path.join(basedir, 'config.json')
        config = Params.from_file(config_path)
        model = Model.load(
            config=config,
            serialization_dir=basedir
        )
        reader = DatasetReader.from_params(config.get('dataset_reader'))
        predictor = Predictor(model=model, dataset_reader=reader)
        return reader, predictor

    @staticmethod
    def _predict_sentence(reader, predictor, sentence, text):
        tokens = []
        for item in sentence:
            if 'text' in item:
                token = item['text'].strip()
            else:
                token = text[item['start']:item['end']].strip()
            if token != '':
                tokens.append(Token(token))
        instance = reader.text_to_instance(tokens)
        return predictor.predict_instance(instance)

    def run(self, request, response):
        lang = request['lang']
        if lang in self.predictors:
            reader = self.readers[lang]
            predictor = self.predictors[lang]
            text = request['text']
            parsing = response['parse']
            debug = request.get('debug', False)
            result = []
            offset = 0
            for sentence in parsing:
                prediction = self._predict_sentence(reader, predictor, sentence, text)
                result = result + _to_annotations(prediction, text, offset=offset, debug=debug)
                offset = sentence[-1]['end']
            return result
        else:
            raise MissingLanguage(lang)

    def describe(self):
        result = super().describe()
        result['langs'] = list(self.descriptions.keys())
        result['models'] = self.descriptions
        return result

class Sentiment(Service):
    def __init__(self, models_dir='models/allen/sentiment'):
        Service.__init__(self, 'sentiment', 'allen-classification', ['parse'])
        self.models = {}
        self.sentiment_maps = {}
        self.indexer = ELMoTokenCharactersIndexer()
        self.descriptions = {}
        for lang in os.listdir(models_dir):
            if len(lang) == 2:
                model, sentiment_map = self._load_model(os.path.join(models_dir, lang))
                self.models[lang] = model
                self.sentiment_maps[lang] = sentiment_map
                self.descriptions[lang] = _load_model_description(os.path.join(models_dir, lang))

    def _load_model(self, basedir):
        config_path = os.path.join(basedir, 'config.json')
        config = Params.from_file(config_path)
        model = Model.load(
            config=config,
            serialization_dir=basedir
        )
        sentiment_map = self._vocab_to_sentiment_map(model.vocab)
        return model, sentiment_map

    @staticmethod
    def _vocab_to_sentiment_map(vocab):
        index_to_token_vocabulary = vocab.get_index_to_token_vocabulary(namespace='labels')
        return {i: float(l) / (len(index_to_token_vocabulary) - 1) for i, l in index_to_token_vocabulary.items() }

    def _predict_sentence(self, model, sentiment_map, sentence, text, debug):
        tokens = []
        for item in sentence:
            if 'text' in item:
                token = item['text'].strip()
            else:
                token = text[item['start']:item['end']].strip()
            if token != '':
                tokens.append(Token(token))
        instance = Instance({
            'tokens': TextField(tokens, token_indexers={'tokens': self.indexer})
        })
        prediction = model.forward_on_instance(instance)
        logits = prediction['logits']
        sentiment = sentiment_map[logits.argmax()]
        start = sentence[0]['start']
        end = sentence[-1]['end']
        item = {
            'start': start,
            'end': end,
            'sentiment': sentiment
        }
        if debug:
            item['text'] = text[start:end]
            item['logits'] = logits.astype('float').tolist()
        return item

    def run(self, request, response):
        lang = request['lang']
        if lang in self.models:
            model = self.models[lang]
            sentiment_map = self.sentiment_maps[lang]
            text = request['text']
            parsing = response['parse']
            debug = request.get('debug', False)
            result = []
            offset = 0
            for sentence in parsing:
                prediction = self._predict_sentence(model, sentiment_map, sentence, text, debug)
                result.append(prediction)
                offset = sentence[-1]['end']
            return result
        else:
            raise MissingLanguage(lang)

    def describe(self):
        result = super().describe()
        result['langs'] = list(self.descriptions.keys())
        result['models'] = self.descriptions
        return result

class SentimentRegression(Service):
    def __init__(self, models_dir='models/allen/sentiment-regression'):
        Service.__init__(self, 'sentiment', 'allen-regression', ['parse'])
        self.models = {}
        self.descriptions = {}
        self.indexer = ELMoTokenCharactersIndexer()
        for lang in os.listdir(models_dir):
            if len(lang) == 2:
                self.models[lang] = self._load_model(os.path.join(models_dir, lang))
                self.descriptions[lang] = _load_model_description(os.path.join(models_dir, lang))

    @staticmethod
    def _load_model(basedir):
        config_path = os.path.join(basedir, 'config.json')
        config = Params.from_file(config_path)
        model = Model.load(
            config=config,
            serialization_dir=basedir
        )
        return model

    def _predict_sentence(self, model, sentence, text, debug):
        tokens = []
        for item in sentence:
            if 'text' in item:
                token = item['text'].strip()
            else:
                token = text[item['start']:item['end']].strip()
            if token != '':
                tokens.append(Token(token))
        instance = Instance({
            'tokens': TextField(tokens, token_indexers={'tokens': self.indexer})
        })
        prediction = model.forward_on_instance(instance)
        sentiment = prediction['sentiment'][0]
        start = sentence[0]['start']
        end = sentence[-1]['end']
        item = {
            'start': start,
            'end': end,
            'sentiment': float(sentiment)
        }
        if debug:
            item['text'] = text[start:end]
        return item

    def run(self, request, response):
        lang = request['lang']
        if lang in self.models:
            model = self.models[lang]
            text = request['text']
            parsing = response['parse']
            debug = request.get('debug', False)
            result = []
            offset = 0
            for sentence in parsing:
                prediction = self._predict_sentence(model, sentence, text, debug)
                result.append(prediction)
                offset = sentence[-1]['end']
            return result
        else:
            raise MissingLanguage(lang)

    def describe(self):
        result = super().describe()
        result['langs'] = list(self.descriptions.keys())
        result['models'] = self.descriptions
        return result
