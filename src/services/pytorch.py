import os
import json

import torch
from torch.autograd import Variable

from services import Service, MissingLanguage
from common.pytorch.ner.model import Tagger


def _lines(path):
    with open(path) as f:
        return [l[:-1] for l in f.readlines()]


def _clean_tag(tag):
    if tag[:2] in ['I-', 'B-']:
        return tag[2:]
    else:
        return tag

class Ner(Service):
    def __init__(self, langs):
        Service.__init__(self, 'ner', 'pytorch', ['parse'])
        model_dir = os.path.join('models', 'pytorch', 'ner')
        self.models = {}
        self.word_indices = {}
        self.word_indices_lower = {}
        self.tags = {}
        self.results = {}
        for lang in langs:
            word_path = os.path.join(model_dir, f'{lang}-words.index')
            tag_path = os.path.join(model_dir, f'{lang}-tags.index')
            model_path = os.path.join(model_dir, f'{lang}.pth')
            results_path = os.path.join(model_dir, f'{lang}-results.json')
            words = _lines(word_path)
            self.word_indices[lang] = {w: i for i, w in enumerate(words)}
            self.word_indices_lower[lang] = {w.lower(): i for i, w in enumerate(words)}
            self.tags[lang] = _lines(tag_path)
            model = torch.load(model_path, map_location='cpu')
            model.device = 'cpu'
            self.models[lang] = model
            self.results[lang] = {'pretrained': False}
            try:
                with open(results_path) as f:
                    self.results[lang].update(json.load(f))
            except:
                # No results found
                pass

    def run(self, request, response):
        lang = request['lang']
        if lang in self.models:
            text = request['text']
            parsing = response['parse']
            debug = request.get('debug', False)
            model = self.models[lang]
            word_index = self.word_indices[lang]
            tags = self.tags[lang]
            SOS = word_index['<SOS>']
            EOS = word_index['<EOS>']

            result = []
            for sentence in parsing:
                tensor = torch.LongTensor(2 + len(sentence))
                tensor[0] = SOS
                tensor[-1] = EOS
                for i, t in enumerate(sentence):
                    # First, find the text of the current token
                    if 'text' in t:
                        token = t['text']
                    else:
                        token = text[t['start']:t['end']]

                    # Then find the index of the token, if any
                    tensor[1+i] = word_index.get(token, 0)

                _, tag_nums = model(Variable(tensor))

                # TODO: clean up regrouping logic
                last_tag = 'O'
                last_start = sentence[0]['start']
                for i in range(len(tag_nums.data) - 1):
                    tag = _clean_tag(tags[tag_nums.data[i]])
                    if tag != last_tag:
                        if not last_tag in ['O', '<SOS>', '<EOS>']:
                            t = sentence[i-1]
                            end = t['start'] - 1
                            if debug:
                                result.append({
                                    'start': last_start,
                                    'end': end,
                                    'label': last_tag,
                                    'text': text[last_start:end]
                                })
                            else:
                                result.append({
                                    'start': last_start,
                                    'end': end,
                                    'label': last_tag
                                })
                        if i > 0:
                            last_tag = tag
                            last_start = sentence[i-1]['start']
                if not last_tag in ['O', '<SOS>', '<EOS>']:
                    end = sentence[-1]['end']
                    if debug:
                        result.append({
                            'start': last_start,
                            'end': end,
                            'label': last_tag,
                            'text': text[last_start:end]
                        })
                    else:
                        result.append({
                            'start': last_start,
                            'end': end,
                            'label': last_tag
                        })

            return result
        else:
            raise MissingLanguage(lang)

    def describe(self):
        result = super().describe()
        result['langs'] = list(self.models.keys())
        result['models'] = {}
        for lang, outcome in self.results.items():
            result['models'][lang] = outcome
        return result