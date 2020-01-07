import os
from collections import defaultdict
from itertools import combinations
from heapq import nlargest

import networkx

from services import Service, MissingLanguage

def _read_lines(path):
    result = []
    with open(path) as f:
        for line in f.readlines():
            result.append(line.strip())
    return result

def _read_stopwords(stopwords_dir):
    result = {}
    for file in os.listdir(stopwords_dir):
        lang, _ = os.path.splitext(file)
        path = os.path.join(stopwords_dir, file)
        result[lang] = _read_lines(path)
    return result

def _parsing_to_tokens(parsing, text, keep, normalize):
    sentences = []
    for p in parsing:
        sentence = []
        for t in p:
            token = normalize(text[t['start']:t['end']])
            if keep(token):
                sentence.append(token)
        sentences.append(sentence)
    return sentences

def _sentence_graph(sentences):
    graph = networkx.Graph()

    for i in range(len(sentences)):
        graph.add_node(i)

    word_to_sentence = defaultdict(set)

    for i, sentence in enumerate(sentences):
        for token in sentence:
            word_to_sentence[token].add(i)

    for s in word_to_sentence.values():
        for i, j in combinations(s, 2):
            graph.add_edge(i, j)
            graph.add_edge(j, i)

    return graph

def _word_graph(sentences):
    graph = networkx.Graph()

    words = set(t for s in sentences for t in s)
    graph.add_nodes_from(words)

    for sentence in sentences:
        for i, j in combinations(sentence, 2):
            graph.add_edge(i, j)
            graph.add_edge(j, i)

    return graph

class Summarize(Service):
    def __init__(self, stopwords_dir=os.path.join('resources', 'stopwords')):
        Service.__init__(self, 'extractive-summarization', 'textrank', ['parse'])
        self.stopwords = _read_stopwords(stopwords_dir)

    def run(self, request, response):
        lang = request['lang']
        if lang in self.stopwords:
            # Extract request data
            text = request['text']
            parsing = response['parse']
            debug = request.get('debug', False)
            num_sentences = request.get('num-extractive-sentences', 3)

            # Cleaning and normalization
            stopwords = self.stopwords[lang]
            keep = lambda x: any(char.isalpha() or char.isdigit() for char in x) \
                and (len(x) > 2) and not (x in stopwords)
            normalize = lambda x: x.lower()

            # Create the sentence graph and compute pagerank
            sentences = _parsing_to_tokens(parsing, text, keep, normalize)
            graph = _sentence_graph(sentences)
            weights = networkx.pagerank(graph)

            # Assemble result
            topk = nlargest(num_sentences, weights.items(), key=lambda x: x[1])
            result = []
            for i, _ in topk:
                sentence = parsing[i]
                start = sentence[0]['start']
                end = sentence[-1]['end']
                if debug:
                    result.append({
                        'start': start,
                        'end': end,
                        'text': text[start:end]
                    })
                else:
                    result.append({
                        'start': start,
                        'end': end
                    })

            return result
        else:
            raise MissingLanguage(lang)

    def describe(self):
        result = super().describe()
        result['langs'] = list(self.stopwords.keys())
        result['extra-params'] = [{
            'name': 'num-extractive-sentences',
            'type': 'int',
            'required': False
        }]
        return result

class Keywords(Service):
    def __init__(self, stopwords_dir=os.path.join('resources', 'stopwords')):
        Service.__init__(self, 'keywords', 'textrank', ['parse'])
        self.stopwords = _read_stopwords(stopwords_dir)

    def run(self, request, response):
        lang = request['lang']
        if lang in self.stopwords:
            # Extract request data
            text = request['text']
            parsing = response['parse']
            num_keywords = request.get('num-keywords', 3)

            # Cleaning and normalization
            stopwords = self.stopwords[lang]
            keep = lambda x: any(char.isalpha() or char.isdigit() for char in x) \
                and (len(x) > 2) and not (x in stopwords)
            normalize = lambda x: x.lower()

            # Create the sentence graph and compute pagerank
            sentences = _parsing_to_tokens(parsing, text, keep, normalize)
            graph = _word_graph(sentences)
            weights = networkx.pagerank(graph)

            # Assemble result
            topk = nlargest(num_keywords, weights.items(), key=lambda x: x[1])
            result = [{'text': t} for t, _ in topk]

            return result
        else:
            raise MissingLanguage(lang)

    def describe(self):
        result = super().describe()
        result['langs'] = list(self.stopwords.keys())
        result['extra-params'] = [{
            'name': 'num-keywords',
            'type': 'int',
            'required': False
        }]
        return result