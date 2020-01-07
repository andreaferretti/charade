from unittest import TestCase

from services.textrank import _word_graph, _sentence_graph


_sentences = [
    ['Every', 'breath',  'you', 'take'],
    ['Every', 'move',  'you', 'make'],
    ['Every', 'bond', 'you', 'break'],
    ['Every', 'step',  'you', 'take'],
    ['I', 'll', 'be', 'watching', 'you']
]

class TestGraphCreation(TestCase):
    def test_word_graph(self):
        g = _word_graph(_sentences)
        self.assertEqual(len(g.nodes), 13)
        self.assertEqual(
            set(g.neighbors('Every')),
            {'bond', 'break', 'breath', 'make', 'move', 'step', 'take', 'you'}
        )

    def test_sentence_graph(self):
        g = _sentence_graph(_sentences)
        self.assertEqual(len(g.nodes), 5)
        self.assertEqual(
            set(g.edges),
            {(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)}
        )