import os
import json
from argparse import Namespace

import onmt
from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus
from onmt.translate.translator import build_translator
from onmt.translate.translator import Translator
import onmt.opts as opts
from onmt.utils.parse import ArgumentParser

from services import Service, MissingLanguage


def _load_model_description(model_dir):
    '''
    Reads configuration parameters and training metrics from JSON files stored with the AllenNLP model.
    Returns a dictionary that can be used to populate the 'models' field of the service description.
    '''
    desc = {}
    try:
        descr_fn = os.path.join(model_dir, 'model_params.json')
        if os.path.isfile(descr_fn):
            with open(descr_fn) as descr_file:
                config = json.load(descr_file)
                desc['datasets'] = config.get('datasets')
                desc['params'] = config.get('params')
                desc['training-time'] = config.get('training_time')
                desc['metrics'] = config.get('metrics')
    except (IOError, ValueError):
        pass
    return desc

class Summarization(Service):
    def __init__(self, models_dir='models/opennmt/summarization'):
        Service.__init__(self, 'abstractive-summarization', 'opennmt', ['parse'])
        # define opt values for the summarisation task
        self.models = {}
        self.descriptions = {}
        for lang in os.listdir(models_dir):
            if len(lang) == 2:
                self.models[lang] = self._load_model(os.path.join(models_dir, lang))
                self.descriptions[lang] = _load_model_description(os.path.join(models_dir, lang))

    @staticmethod
    def _load_model(basedir):
        with open(os.path.join(basedir,'config_dict.json'), 'r') as fp:
            baseopt = json.load(fp)

        opt = Namespace(**baseopt)
        opt.models = [os.path.join(basedir, 'cnndm_step_200000.pt')]
        opt.gpu = -1
        translator = build_translator(opt, report_score=True)

        return translator

    def _summarize_text(self, model, text, debug):
        score, pred = model.translate(src=[text], batch_size=20)
        summary = (pred[0])[0].replace(' .', '.').replace('<t> ', '').replace(' </t>', '').strip()
        return {
            'text': summary,
            'score': (score[0][0]).item()
        }

    def run(self, request, response):
        lang = request['lang']
        if lang in self.models:
            model = self.models[lang]
            text = request['text']
            parsing = response['parse']
            debug = request.get('debug', False)

            sentences = []
            for sentence in parsing:
                tokens = []
                for item in sentence:
                    if 'text' in item:
                        token = item['text'].strip()
                    else:
                        token = text[item['start']:item['end']].strip()
                    if token != '':
                        tokens.append(token)
                sentences.append('<t> ' + ' '.join(tokens) + ' </t>')
            summary_response = self._summarize_text(model,' '.join(sentences),debug)

            item = {
                'summary': summary_response['text']
            }

            if debug:
                item['summarization_ratio'] = len(summary_response['text']) / len(text)
                item['prediction_score'] = summary_response['score']

            return item
        else:
            raise MissingLanguage(lang)

    def describe(self):
        result = super().describe()
        result['langs'] = list(self.descriptions.keys())
        result['models'] = self.descriptions
        return result

class Translation(Service):
    def __init__(self, models_dir='models/opennmt/translation'):
        Service.__init__(self, 'translation', 'opennmt', ['parse'])
        # define opt values for the summarisation task
        self.models = {}
        self.descriptions = {}
        for lang in os.listdir(models_dir):
            if len(lang) == 5:
                self.models[lang] = self._load_model(os.path.join(models_dir, lang),lang)
                self.descriptions[lang] = _load_model_description(os.path.join(models_dir, lang))

    @staticmethod
    def _load_model(basedir,lang):
        with open(os.path.join(basedir,'config_dict.json'), 'r') as fp:
            baseopt = json.load(fp)

        opt = Namespace(**baseopt)
        opt.models = [os.path.join(basedir, lang + '_step_200000.pt')]
        opt.gpu = -1
        translator = build_translator(opt, report_score=True)

        return translator

    def _translate_text(self, model, text, debug):
        score, pred = model.translate(src=[text], batch_size=20)
        summary = (pred[0])[0].replace(' .', '.').replace('<t> ', '').replace(' </t>', '').strip()
        return {
            'text': summary,
            'score': (score[0][0]).item()
        }

    def run(self, request, response):
        lang_pair = request['lang'] + '-' + request['target-lang']
        trl_sentences = []
        scores = []
        sentencewise = True

        if lang_pair in self.models:
            model = self.models[lang_pair]
            text = request['text']
            parsing = response['parse']
            debug = request.get('debug', False)
            sentences = []
            for sentence in parsing:
                tokens = []
                for item in sentence:
                    if 'text' in item:
                        token = item['text'].strip()
                    else:
                        token = text[item['start']:item['end']].strip()
                    if token != '':
                        tokens.append(token)
                if sentencewise:
                    trl_output = self._translate_text(model,' '.join(tokens),debug)
                    trl_sentences.append(trl_output['text'])
                    if debug:
                        scores.append(trl_output['score'])
                else:
                    sentences.append(' '.join(tokens))

            # Here translate sentence by sentence (otherwise it seems that it will just translate + summarize)
            item = {}

            if sentencewise:
                item['translation'] = '  '.join(trl_sentences)
                if debug:
                    #item['summarization_ratio'] = len(summary_response['text'])/len(text)
                    item['prediction_score'] = scores
            else:
                trl_output = self._translate_text(model, ' '.join(sentences), debug)
                item['translation'] = trl_output['text']
                item['prediction_score'] = trl_output['score']

            return item
        else:
            raise MissingLanguage(lang_pair)

    def describe(self):
        result = super().describe()
        result['langs'] = list(x[:2] for x in self.descriptions.keys())
        result['models'] = self.descriptions
        result['extra-params'] = [{
            'name': 'target-lang',
            'type': 'string',
            'required': True
        }]
        return result