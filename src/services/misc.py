import os
from string import ascii_uppercase
from datetime import datetime

from dateparser.search import search_dates

from services import Service, MissingLanguage


_format = '%Y-%m-%d'
_vowels = 'AEIOU'
_consonants = ''.join(c for c in ascii_uppercase if not c in _vowels)

class Dates(Service):
    def __init__(self, langs):
        Service.__init__(self, 'dates', 'misc', [])
        self.langs = langs

    def run(self, request, response):
        lang = request['lang']
        if lang in self.langs:
            text = request['text']
            debug = request.get('debug', False)
            start = 0
            end = 0
            result = []
            for chunk, date in search_dates(text, languages=self.langs):
                start = text.index(chunk, end)
                end = start + len(chunk)
                if debug:
                    result.append({
                        'text': chunk,
                        'start': start,
                        'end': end,
                        'date': date.strftime(_format)
                    })
                else:
                    result.append({
                        'start': start,
                        'end': end,
                        'date': date.strftime(_format)
                    })
            return result
        else:
            raise MissingLanguage(lang)

    def describe(self):
        result = super().describe()
        result['langs'] = self.langs
        return result

def _split_chars(s):
    s = s.upper()
    vowels, consonants = '', ''
    for c in s:
        if c in _vowels:
            vowels += c
        elif c in _consonants:
            consonants += c
    return vowels, consonants

def _cf_name(name, surname):
    surname_vowels, surname_consonants = _split_chars(surname)
    name_vowels, name_consonants = _split_chars(name)
    cf = (surname_consonants + surname_vowels + 'XXX')[:3]
    if len(name_consonants) >= 4:
        cf += name_consonants[0] + name_consonants[2] + name_consonants[3]
    else:
        cf += name_consonants + name_vowels + 'XXX'
    return cf[:6]

def _fix_case(tokens):
    return ' '.join(t.capitalize() for t in tokens)

def _line_set(path):
    result = []
    with open(path) as f:
        for line in f.readlines():
            result.append(line.strip())
    return set(result)

# Only italian for now
class Names(Service):
    def __init__(self):
        Service.__init__(self, 'names', 'misc', ['ner'], ['fiscal_code'])
        self._person_prefixes = ['sig.ra', 'sig.a', 'sig.na', 'sig', 'sig.', 'avv', 'avv.', 'dott', 'dott.', 'dr', 'dr.', 'egr', 'ra']
        self._names = _line_set(os.path.join('resources', 'names', 'it.txt'))
        self._surnames = _line_set(os.path.join('resources', 'surnames', 'it.txt'))

    def split_name(self, s, fiscal_codes):
        tokens = s.lower().split()
        tokens = [t for t in tokens if not t in self._person_prefixes]
        if len(tokens) == 0:
            return '', ''
        fiscal_codes_starts = {x[:6] for x in fiscal_codes}
        for i in range(1, len(tokens)):
            name = ' '.join(tokens[:i])
            surname = ' '.join(tokens[i:])
            if _cf_name(name, surname) in fiscal_codes_starts:
                return _fix_case(tokens[:i]), _fix_case(tokens[i:])
            elif _cf_name(surname, name) in fiscal_codes_starts:
                return _fix_case(tokens[i:]), _fix_case(tokens[:i])
        name_tokens = []
        surname_tokens = []
        need_to_reverse = not (tokens[0] in self._names)
        if need_to_reverse:
            tokens.reverse()
        count = 0
        for t in tokens:
            if t in self._names:
                name_tokens.append(t)
                count += 1
            else:
                break
        surname_tokens = tokens[count:]

        # for t in surname_tokens:
        #     if not t in self._surnames:
        #         logging.info('Unkonwn surname: %s', t)
        if need_to_reverse:
            name_tokens.reverse()
            surname_tokens.reverse()

        # Fix De Rosa Carmela and similar
        if len(surname_tokens) == 1 and len(name_tokens) > 0 and surname_tokens[0] in ['de', 'di', 'del']:
            surname_tokens.append(name_tokens[0])
            name_tokens = name_tokens[1:]
        return _fix_case(name_tokens), _fix_case(surname_tokens)

    def run(self, request, response):
        lang = request['lang']
        if lang == 'it':
            text = request['text']
            debug = request.get('debug', False)
            result = []
            for entity in response['ner']:
                if entity['label'] == 'PER':
                    if 'text' in entity:
                        s = entity['text']
                    else:
                        s = text[entity['start']: entity['end']]
                    fiscal_codes = []
                    if 'fiscal_code' in response:
                        fiscal_codes = [x['text'] for x in response['fiscal_code']]
                    name, surname = self.split_name(s, fiscal_codes)
                    person = {
                        'start': entity['start'],
                        'end': entity['end'],
                        'name': name,
                        'surname': surname
                    }
                    result.append(person)
            return result
        else:
            raise MissingLanguage(lang)

    def describe(self):
        result = super().describe()
        result['langs'] = ['it']
        return result

class FiscalCode(Service):
    def __init__(self):
        Service.__init__(self, 'fiscal-code', 'misc', ['codes'])
        self._months = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'H': 6, 'L': 7, 'M': 8, 'P': 9, 'R': 10, 'S': 11, 'T': 12}
        self._odd = {'0': 1,'1': 0,'2': 5,'3': 7,'4': 9,'5': 13,'6': 15,'7': 17,'8': 19,
                     '9': 21,'A': 1,'B': 0,'C': 5,'D': 7,'E': 9,'F': 13,'G': 15,'H': 17,
                     'I': 19,'J': 21,'K': 2,'L': 4,'M': 18,'N': 20,'O': 11,'P': 3,'Q': 6,
                     'R': 8,'V': 10,'S': 12,'T': 14,'U': 16,'W': 22,'X': 25,'Y': 24,'Z': 23}
        self._even ={'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
                     '9': 9, 'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7,
                     'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15,'Q': 16,
                     'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25}

    @staticmethod
    def guess_year(yy):
        yy = int(yy)
        this_year = datetime.now().year
        yyyy = 2000 + yy
        if yyyy < this_year:
            return 2000 + yy
        else:
            return yyyy - 100

    def check_fiscal_code(self, cf):
        cf_, last_char = cf[:-1], cf[-1]
        odds = [self._odd[c] for c in cf_[0::2]]
        evens = [self._even[c] for c in cf_[1::2]]
        control_char = ascii_uppercase[(sum(odds) + sum(evens)) % 26]
        return last_char == control_char

    def birth_date_and_gender(self, cf):
        year = self.guess_year(cf[6:8])
        month = self._months[cf[8]]
        day = int(cf[9:11])
        if day < 40:
            sex = 'M'
        else:
            sex = 'F'
            day -= 40
        date = datetime(year, month, day)
        return {'sex': sex, 'birthdate': date.strftime(_format)}

    def run(self, request, response):
        lang = request['lang']
        if lang == 'it':
            debug = request.get('debug', False)
            result = []
            for code in response['codes']:
                if code['type'] == 'FISCAL_CODE':
                    code_ = dict(code)
                    code_['correct'] = self.check_fiscal_code(code['text'])
                    code_.update(self.birth_date_and_gender(code['text']))
                    result.append(code_)
            return result
        else:
            raise MissingLanguage(lang)

    def describe(self):
        result = super().describe()
        result['langs'] = ['it']
        return result