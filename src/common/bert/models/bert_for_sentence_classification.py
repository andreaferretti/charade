import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
import transformers


import common.bert.logger as logger
import common.bert.constants as constants
import common.bert.dataset_preprocessing as dataset_preprocessing
import common.bert.models_utilities as models_utilities
import common.bert.models.bert_model as bert_model


log = logger.get_logger(logger.INFO)


class BertForSentenceClassification(bert_model.BertModel):
    """
    Bert model for sentence classification task.

    The hyperparameters are specified as parameters of the __init__ method
    (e.g. alpha, batch_size, epochs), together with the specification of the
    pretrained model {pretrained_model_name_or_path} and other configurations
    about the preprocessing ({max_len}, {do_lower_case}).

    Specifically, the pretrained Bert model used is defined by
    {pretrained_model_name_or_path}, e.g. 'bert-base-uncased' or
    the folder containing saved model weights.
    In particular, the preprocessing is performed with the Bert tokenizer:
    tokenization and optionally lower casing, mapping to ids,
    padding to a maximum length (either passed as {max_len}, or
    computed during training on the training set if None is passed);

    During training, the configuration of the dataset is specified by
    a configuration file: the data is specified in the {config} dictionary,
    stored in {path_to_config_folder}: a training set is mandatory, while
    validation and test sets are optional.
    Note that if a validation set is not provided, a random
    splitting is performed over the training set to get actual
    training set and validation set (in order to have evaluation metrics).
    Moreover, the config file also specifies the columns where the sentence
    and the target are stored: respectively in the fields
    {constants.SENTENCE_COLUMN) and {constants.TARGET_COLUMN}.
    Training is performed with Adam optimizer with fixed weight decay.

    Methods:
        self.configure_dataset_loader(configs, path_to_config_folder, valid_size, is_return_datasets = False)
        self.train(dataset_loader, dataset_name='', metrics=[constants.METRICS.ACCURACY],
                   evaluation_phases=[constants.VALID])
        self.predict_dataset(dataset_loader, phase = constants.VALID)
        self.evaluate_dataset(dataset_loader, phase = constants.VALID, metrics = [constants.METRICS.ACCURACY])
        self.save_model(trained_models_folder, model_name, training_info)
        self.predict_example(*args)

    Attributes
    ----------
    language : str
        Language, supported ones are defined in constants.LANGUAGES
    num_labels : int
        Number of labels
    is_single_sentence : boolean
        It is True if the task input is a single sentence, or
        it is False if the task input is a sentences pair
    pretrained_model_name_or_path : str, default 'bert-base-uncased'
        Pretrained Bert model name or path to a directory containing
        saved model weights
    alpha : float, default 1e-5
        Learning rate
    batch_size : int, default 32
        Dimension of batches
    epochs : int, default 4
        Number of epochs (recommended between 2 and 4)
    random_state : int, default 0
        Random state
    max_len : int or None, default None
        If not None, maximum length for sentences, otherwise the
        maximum length is taken from the training dataset during training
    do_lower_case : boolean, default True
        Whether doing the lower casing
    is_cuda : boolean
        Whether cuda is available
    device : torch.device
        Device
    tokenizer : transformers.tokenization_bert.BertTokenizer
        Tokenizer
    model : transformers.modeling_bert.BertForSequenceClassification
        Bert model for sentence classification
    id2label : dict or None, default None
        None or dictionary mapping index to label name; it is
        eventually configured when data is loaded or when model is loaded

    """

    def __init__(self, language, num_labels, pretrained_model_name_or_path='bert-base-uncased',
                 alpha=1e-5, batch_size=32, epochs=4, random_state=0, max_len = None, do_lower_case=True):

        super().__init__(language, True, pretrained_model_name_or_path, alpha, batch_size, epochs, random_state, max_len, do_lower_case)

        self.num_labels = num_labels
        self.model = self._configure_model()

    def _get_model(self):
        """
        Utility function for self._configure_training(): returns
        pretrained Bert model for current task.

        Returns
        -------
        model : transformers.modeling_bert.BertForSequenceClassification
            Bert model

        """

        model = transformers.BertForSequenceClassification.from_pretrained(self.pretrained_model_name_or_path,
                                                                           num_labels=self.num_labels,
                                                                           output_attentions=False,
                                                                           output_hidden_states=False)

        self.id2label = models_utilities.load_labels(self.pretrained_model_name_or_path)

        return model

    @staticmethod
    def _get_tensor_dataset(data, target_column):

        input_ids = torch.tensor(data[constants.TOKEN_IDS].tolist())
        mask = torch.tensor(data[constants.ATT_MASK].tolist())

        y = None
        if target_column in data.columns:
            y = torch.tensor(data[target_column].tolist())

        tensor_dataset = torch.utils.data.TensorDataset(input_ids, mask, y) if y is not None else torch.utils.data.TensorDataset(input_ids, mask)

        return tensor_dataset

    def _batch_forward_propagation(self, batch, phase, is_add_labels):

        batch = tuple(t.to(self.device) for t in batch)

        b_input_ids = batch[0]
        b_input_mask = batch[1]
        b_labels = batch[-1] if is_add_labels else None

        # forward propagation
        if phase == constants.TRAIN:
            self.model.zero_grad()
            outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        else:
            with torch.no_grad():
                outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

        return outputs, b_labels

    def _evaluate(self, logits, labels, metric):

        logits = logits.detach().cpu().numpy()
        labels = labels.cpu().numpy()

        y_pred = np.argmax(logits, axis=1).flatten()
        y = labels.flatten()

        average = 'binary' if self.num_labels == 2 else 'weighted'
        metric_result = models_utilities.evaluate(y, y_pred, metric, average)

        return metric_result

    def predict_example(self, sentence):

        # data preprocessing
        data = pd.DataFrame({'sentence': [sentence]})
        dataset_preprocessing.preprocess_dataset_single_sentence(data, 'sentence', self.max_len, self.tokenizer, self.language)

        # to tensor
        input_ids = torch.tensor(data[constants.TOKEN_IDS].tolist()).to(self.device)
        mask = torch.tensor(data[constants.ATT_MASK].tolist()).to(self.device)

        # forward propagation
        with torch.no_grad():
            outputs = self.model(input_ids, token_type_ids=None, attention_mask=mask)

        # prediction
        logits = outputs[0]
        probabilities = F.softmax(logits, dim=1)
        probabilities = probabilities.detach().cpu().numpy()[0].tolist()

        logits = logits.detach().cpu().numpy()
        y_pred = np.argmax(logits, axis=1).flatten().tolist()[0]

        return y_pred, probabilities


