import re
from collections import defaultdict

from services import Service, MissingLanguage


class Parse(Service):
    def __init__(self):
        Service.__init__(self, 'parse', 'regex', [])
        self.sentence_regex = re.compile(r'([.!?]\s+|$)')
        self.token_regex = re.compile(r'([\s;,:.!?]|$)')

    def run(self, request, response):
        text = request['text']
        debug = request.get('debug', False)
        result = []
        sentence_start = 0
        sentence_end = 0
        for punctuation in self.sentence_regex.finditer(text):
            s, e = punctuation.span()
            sentence_end = e
            sentence = text[sentence_start:sentence_end]
            token_start = sentence_start
            token_end = sentence_start
            tokens = []
            for spaces in self.token_regex.finditer(sentence):
                s_, e_ = spaces.span()
                token_end = sentence_start + s_
                # Only append non-trivial tokens
                if token_end > token_start:
                    if debug:
                        tokens.append({
                            'text': text[token_start:token_end],
                            'start': token_start,
                            'end': token_end
                        })
                    else:
                        tokens.append({
                            'start': token_start,
                            'end': token_end
                        })
                token_start = sentence_start + e_
                # Consider punctuation as its own tokens
                token_char = sentence[s_:s_+1]
                if token_char and not token_char.isspace():
                    if debug:
                        tokens.append({
                            'text': token_char,
                            'start': sentence_start + s_,
                            'end': sentence_start + s_ + 1
                        })
                    else:
                        tokens.append({
                            'start': sentence_start + s_,
                            'end': sentence_start + s_ + 1
                        })
            if len(tokens) > 0:
                result.append(tokens)
            sentence_start = e
        return result

    def describe(self):
        result = super().describe()
        result['langs'] = ['*']  # this service supports most latin-based languages
        return result

class Codes(Service):
    def __init__(self):
        Service.__init__(self, 'codes', 'regex', [])
        self.regexes = [
            (re.compile(r'([a-z]{6}\s?\d{2}\s?[a-z]{1}\s?\d{2}\s?[a-z]{1}\s?\d{3}\s?[a-z]{1})', re.IGNORECASE), 'FISCAL_CODE', 'it'),
            (re.compile(r'(IT\d{2}[ ][a-zA-Z]\d{3}[ ]\d{4}[ ]\d{4}[ ]\d{4}[ ]\d{4}[ ]\d{3})', re.IGNORECASE), 'IBAN', 'it'),
            (re.compile(r'(IT\d{2}[a-zA-Z]\d{22}|IT\d{2}[a-zA-Z][ ]\d{5}[ ]\d{5}[ ]\d{12})', re.IGNORECASE), 'IBAN', 'it'),
            (re.compile(r'(IT\s?\d{2}\s?[a-z]\s?\d{8}\s?\d{6}\s?\d{8})', re.IGNORECASE), 'IBAN', 'it'),
            (re.compile(r'\D(\d{11})\D', re.IGNORECASE), 'PIVA', 'it')
        ]

    @staticmethod
    def make_value(m):
        return {'start': m.start(), 'end': m.end(), 'text': m.groups(0)[0].upper()}

    @staticmethod
    def remove_duplicates(codes):
        res = dict()
        for code in codes:
            res[(code['text'], code['start'])] = code
        return sorted(res.values(), key=lambda x: x['start'])

    def run(self, request, response):
        text = request['text']
        debug = request.get('debug', False)
        result = []
        for code_regex, code_type, code_lang in self.regexes:
            for match in code_regex.finditer(text):
                code = self.make_value(match)
                code['type'] = code_type
                code['lang'] = code_lang
                result.append(code)
        return self.remove_duplicates(result)

    def describe(self):
        result = super().describe()
        # extract the language from self.regexes and remove duplicates
        result['langs'] = list({code_lang for _, _, code_lang in self.regexes})
        return result
