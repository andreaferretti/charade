import nltk
from nltk.tokenize import TreebankWordTokenizer, PunktSentenceTokenizer
from nltk.chunk import ne_chunk, tree2conlltags

from services import Service, MissingLanguage


def _annotation2token(annotation, text):
    if 'text' in annotation:
        return annotation['text']
    else:
        return text[annotation['start']:annotation['end']]

def _conll(tokens):
    pos_tags = nltk.pos_tag(tokens)
    named_entities = ne_chunk(pos_tags)
    return [(x[0], x[2]) for x in tree2conlltags(named_entities)]

class Ner(Service):
    def __init__(self, langs=[]):
        Service.__init__(self, 'ner', 'nltk', ['parse'])

    def run(self, request, response):
        if request['lang'] == 'en':
            text = request['text']
            parsing = response['parse']
            debug = request.get('debug', False)
            result = []
            for sentence in parsing:
                tokens = [_annotation2token(t, text) for t in sentence]
                tags = _conll(tokens)
                for ann, (token, tag) in zip (sentence, tags):
                    if tag.startswith('B-'):
                        item = {
                            'start': ann['start'],
                            'end': ann['end'],
                            'label': tag.partition('-')[2]
                        }
                        if debug:
                            item['text'] = ann['text']
                        result.append(item)
                    elif tag.startswith('I-'):
                        item = result[-1]
                        item['end'] = ann['end']
                        if debug:
                            item['text'] = text[item['start']:item['end']]
            return result
        else:
            raise MissingLanguage(request['lang'])

    def describe(self):
        result = super().describe()
        result['langs'] = ['en']
        result['models'] = {'en': {'pretrained': True}}
        return result

class Parse(Service):
    def __init__(self, langs=[]):
        Service.__init__(self, 'parse', 'nltk', [])
        self.punktSentenceTokenizer = PunktSentenceTokenizer()
        self.treebankWordTokenizer = TreebankWordTokenizer()
        #PunkSentence, TreebankWordTokenizer

    def run(self, request, response):
        if request['lang'] == 'en':
            text = request['text']
            debug = request.get('debug', False)
            result = []
            for sent_s, sent_e in self.punktSentenceTokenizer.span_tokenize(text):
                tokens = []
                sentence = text[sent_s:sent_e]
                for token_s, token_e in self.treebankWordTokenizer.span_tokenize(sentence):
                    item = {
                        'start': token_s + sent_s,
                        'end': token_e + sent_s
                    }
                    if debug:
                        item['text'] = sentence[token_s:token_e]
                    tokens.append(item)
                result.append(tokens)
            return result
        else:
            raise MissingLanguage(request['lang'])

    def describe(self):
        result = super().describe()
        result['langs'] = ['en']
        result['models'] = {'en': {'pretrained': True}}
        return result

