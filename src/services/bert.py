# References: https://mccormickml.com/2019/07/22/BERT-fine-tuning/
# for Bert in Italian language: http://ceur-ws.org/Vol-2481/paper57.pdf

import os

import transformers
import json
import common.bert.models.bert_for_sentence_classification as bert_for_sentence_classification
import common.bert.models.bert_for_next_sentence_prediction as bert_for_next_sentence_prediction
import common.bert.constants as constants
import common.bert.models_utilities as models_utilities



from services import Service, MissingModel, MissingParameter, MissingResource



class SentenceClassification(Service):

    def __init__(self, model_dir=os.path.join('models', 'bert', 'classification')):

        Service.__init__(self, 'classification', 'bert', ['parse'])
        self.models = {}
        self.results = {}
        self.id2label = {}
        langs = set()
        for name in os.listdir(model_dir):
            if not os.path.isdir(os.path.join(model_dir, name)):
                continue

            with open(os.path.join(model_dir, name, transformers.CONFIG_NAME), 'r') as f:
                configs = json.load(f)

            num_labels = configs['_num_labels']
            language = configs[constants.MODEL_INFO][constants.LANGUAGE]
            langs.add(language)

            pretrained_model_name_or_path = os.path.join(model_dir, name)
            self.models[name] = bert_for_sentence_classification.BertForSentenceClassification(language, num_labels,
                                                                                               pretrained_model_name_or_path)
            self.results[name] = configs[constants.MODEL_INFO]
            self.id2label[name] = models_utilities.load_labels(pretrained_model_name_or_path)

        self.langs = list(langs)


    def _predict_sentence(self, model, sentence, text, id2label, debug):

        start = sentence[0]['start']
        end = sentence[-1]['end']

        predicted_label, probabilities = model.predict_example(text[start:end])

        item = {
            'start': start,
            'end': end,
            'label': predicted_label if id2label is None else id2label[str(predicted_label)]
        }
        if debug:
            item['text'] = text[start:end]
            item['probabilities'] = probabilities

        return item


    def run(self, request, response):
        if 'bert-model-classification' in request:
            model_name = request['bert-model-classification']
        else:
            raise MissingParameter(self.task, self.name, 'bert-model-classification')

        if model_name in self.models:
            model = self.models[model_name]
            id2label = self.id2label[model_name]
        else:
            raise MissingModel(self.task, self.name, model_name, list(self.models))

        lang = request['lang']
        if lang not in self.langs:
            raise MissingResource(self.task, self.name, f'{lang}.txt')

        text = request['text']
        parsing = response['parse']
        debug = request.get('debug', False)
        result = []
        for sentence in parsing:
            prediction = self._predict_sentence(model, sentence, text, id2label, debug)
            result.append(prediction)

        return result

    def describe(self):
        result = super().describe()
        result['langs'] = self.langs
        result['extra-params'] = [{
            'name': 'bert-model-classification',
            'type': 'string',
            'choices': list(self.models.keys()),
            'required': True
        }]
        result['models'] = self.results
        return result




class NextSentencePrediction(Service):

    def __init__(self, model_dir=os.path.join('models', 'bert', 'next_sentence_prediction')):

        Service.__init__(self, 'next-sentence-prediction', 'bert', ['parse'])
        self.models = {}
        self.results = {}
        langs = set()
        for name in os.listdir(model_dir):
            if not os.path.isdir(os.path.join(model_dir, name)):
                continue

            with open(os.path.join(model_dir, name, transformers.CONFIG_NAME), 'r') as f:
                configs = json.load(f)

            language = configs[constants.MODEL_INFO][constants.LANGUAGE]
            langs.add(language)

            pretrained_model_name_or_path = os.path.join(model_dir, name)
            self.models[name] = bert_for_next_sentence_prediction.BertForNextSentencePrediction(language, pretrained_model_name_or_path)
            self.results[name] = configs[constants.MODEL_INFO]

        self.langs = list(langs)


    def _predict_sentences_pair(self, model, sentence1, sentence2, text, debug):

        start1 = sentence1[0]['start']
        end1 = sentence1[-1]['end']

        start2 = sentence2[0]['start']
        end2 = sentence2[-1]['end']

        predicted_label, probabilities = model.predict_example(text[start1:end1], text[start2:end2])

        item = {
            'start1': start1,
            'end1': end1,
            'start2': start2,
            'end2': end2,
            'is_next': predicted_label
        }
        if debug:
            item['sentence1'] = text[start1:end1]
            item['sentence2'] = text[start2:end2]
            item['probabilities'] = probabilities

        return item


    def run(self, request, response):
        if 'bert-model-nsp' in request:
            model_name = request['bert-model-nsp']
        else:
            raise MissingParameter(self.task, self.name, 'bert-model-nsp')

        if model_name in self.models:
            model = self.models[model_name]
        else:
            raise MissingModel(self.task, self.name, model_name, list(self.models))

        lang = request['lang']
        if lang not in self.langs:
            raise MissingResource(self.task, self.name, f'{lang}.txt')

        text = request['text']
        parsing = response['parse']
        debug = request.get('debug', False)
        result = []
        if len(parsing) < 2:
            return result

        sentence1 = parsing[0]
        for sentence2 in parsing[1:]:
            prediction = self._predict_sentences_pair(model, sentence1, sentence2, text, debug)
            result.append(prediction)

            sentence1 = sentence2

        return result

    def describe(self):
        result = super().describe()
        result['langs'] = self.langs
        result['extra-params'] = [
            {
                'name': 'bert-model-nsp',
                'type': 'string',
                'choices': list(self.models.keys()),
                'required': True
            }]
        result['models'] = self.results
        return result