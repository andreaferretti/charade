import spacy

from services import Service, MissingLanguage


class Ner(Service):
    def __init__(self, langs):
        Service.__init__(self, 'ner', 'spacy', [])
        self.models = {}
        for lang in langs:
            self.models[lang] = spacy.load(lang)

    def run(self, request, response):
        lang = request['lang']
        if lang in self.models:
            text = request['text']
            debug = request.get('debug', False)
            model = self.models[lang]
            doc = model(text)
            result = []
            for ent in doc.ents:
                if debug:
                    result.append({
                        'text': ent.text,
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'label': ent.label_
                    })
                else:
                    result.append({
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'label': ent.label_
                    })
            return result
        else:
            raise MissingLanguage(lang)

    def describe(self):
        result = super().describe()
        result['langs'] = list(self.models.keys())
        result['models'] = {}
        for lang in result['langs']:
            result['models'][lang] = {
                'pretrained': True
            }
        return result

class Parse(Service):
    def __init__(self, langs):
        Service.__init__(self, 'parse', 'spacy', [])
        self.models = {}
        for lang in langs:
            self.models[lang] = spacy.load(lang)

    def run(self, request, response):
        lang = request['lang']
        if lang in self.models:
            text = request['text']
            debug = request.get('debug', False)
            model = self.models[lang]
            doc = model(text)
            result = []
            for sentence in doc.sents:
                tokens = []
                for token in sentence:
                    start = token.idx
                    end = start + len(token)
                    if debug:
                        tokens.append({
                            'text': token.text,
                            'start': start,
                            'end': end
                        })
                    else:
                        tokens.append({
                            'start': start,
                            'end': end
                        })
                result.append(tokens)
            return result
        else:
            raise MissingLanguage(lang)

    def describe(self):
        result = super().describe()
        result['langs'] = list(self.models.keys())
        result['models'] = {}
        for lang in result['langs']:
            result['models'][lang] = {
                'pretrained': True
            }
        return result