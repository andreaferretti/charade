# All the code in the present section is adapted from
# http://www.realworldnlpbook.com/blog/training-sentiment-analyzer-using-allennlp.html
# http://www.realworldnlpbook.com/blog/improving-sentiment-analyzer-using-elmo.html
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from allennlp.nn.util import get_text_field_mask
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields.array_field import ArrayField


@Model.register('lstm_classifier')
class LstmClassifier(Model):
    def __init__(self,
        word_embeddings: TextFieldEmbedder,
        encoder: Seq2VecEncoder,
        vocab: Vocabulary) -> None:

        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.out = nn.Linear(in_features=encoder.get_output_dim(), out_features=vocab.get_vocab_size('labels'))
        self.accuracy = CategoricalAccuracy()
        self.f1_measure = F1Measure(4)
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self,
            tokens: Dict[str, torch.Tensor],
            label: torch.Tensor = None) -> torch.Tensor:
        mask = get_text_field_mask(tokens)

        embeddings = self.word_embeddings(tokens)
        encoder_out = self.encoder(embeddings, mask)
        logits = self.out(encoder_out)

        output = {'logits': logits}
        if label is not None:
            self.accuracy(logits, label)
            self.f1_measure(logits, label)
            output['loss'] = self.loss_function(logits, label)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        precision, recall, f1_measure = self.f1_measure.get_metric(reset)
        accuracy = self.accuracy.get_metric(reset)
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_measure': f1_measure
        }

@Model.register('lstm_regressor')
class LstmClassifier(Model):
    def __init__(self,
        word_embeddings: TextFieldEmbedder,
        encoder: Seq2VecEncoder,
        vocab: Vocabulary) -> None:

        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.out = nn.Linear(in_features=encoder.get_output_dim(), out_features=1)
        self.loss_function = nn.L1Loss()

    def forward(self,
            tokens: Dict[str, torch.Tensor],
            label: torch.Tensor = None) -> torch.Tensor:
        mask = get_text_field_mask(tokens)

        embeddings = self.word_embeddings(tokens)
        encoder_out = self.encoder(embeddings, mask)
        sentiment = self.out(encoder_out)

        output = {'sentiment': sentiment}
        if label is not None:
            output['loss'] = self.loss_function(sentiment, label)

        return output

@DatasetReader.register("wrapper")
class WrapperReader(DatasetReader):
    def __init__(self, base_reader: DatasetReader, max_label: int = 4, lazy: bool = False):
        super().__init__(lazy)
        self.base_reader = base_reader
        self.max_label = max_label

    def _parse_label(self, instance):
        label = instance.get('label').label
        parsed_label = float(label) / self.max_label
        instance.add_field('label', ArrayField(np.array([parsed_label])))
        return instance

    def _read(self, file_path):
        return (self._parse_label(instance) for instance in self.base_reader._read(file_path))