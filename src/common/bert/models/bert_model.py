import os
import time
import datetime
import random
import json
import logging


import numpy as np

import torch
import transformers

import common.bert.logger as logger
import common.bert.constants as constants
import common.bert.dataset_preprocessing as dataset_preprocessing
import common.bert.models_utilities as models_utilities


log = logger.get_logger(logger.INFO)


logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)


class BertModel():
    """
    Super class of Bert model for generic task, providing common tools
    for training and evaluation.

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
    Moreover, the config file also specifies the columns where sentence(s)
    and target are stored. Respectively, if {is_single_sentence} is True,
    the sentence is stored in the field {constants.SENTENCE_COLUMN), otherwise
    the sentences are stored respectively in fields {constants.SECOND_SENTENCE_COLUMN)
    and {constants.FIRST_SENTENCE_COLUMN). The target is stored in field
    {constants.TARGET_COLUMN}.
    Training is performed with Adam optimizer with fixed weight decay.

    Methods:
        self.configure_dataset_loader(configs, path_to_config_folder, valid_size, is_return_datasets = False)
        self.train(dataset_loader, dataset_name='', metrics=[constants.METRICS.ACCURACY],
                   evaluation_phases=[constants.VALID])
        self.predict_dataset(dataset_loader, phase = constants.VALID)
        self.evaluate_dataset(dataset_loader, phase = constants.VALID, metrics = [constants.METRICS.ACCURACY])
        self.save_model(trained_models_folder, model_name, training_info)
        self.predict_example(*args)

    A sub-class of bert_model.BertModel inside the __init__ method defines
    additional parameters (if any) and then must call the following method:
        self._configure_model()

    Moreover, it must implement the following custom methods
    depending on the specific task:
        _get_model(self): custom Bert model class
        _get_tensor_dataset(self, data, target_column): custom tensor dataset
        _batch_forward_propagation(self, batch, phase, is_add_labels):
        custom forward propagation on given batch
        _evaluate(self, logits, labels, metric): custom task evaluation
        predict_example(self, *args): custom prediction on new example

    Attributes
    ----------
    language : str
        Language, supported ones are defined in constants.LANGUAGES
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
    model : transformers.modeling_bert.*
        Bert model for current task
    id2label : dict or None, default None
        None or dictionary mapping index to label name; it is
        eventually configured when data is loaded or when model is loaded

    """

    def __init__(self, language, is_single_sentence, pretrained_model_name_or_path='bert-base-uncased',
                 alpha=1e-5, batch_size=32, epochs=4, random_state=0, max_len=None, do_lower_case=True):

        self.language = language
        self.is_single_sentence = is_single_sentence
        self.pretrained_model_name_or_path = pretrained_model_name_or_path

        # parameters
        self.alpha = alpha
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state

        # preprocessing configuration
        self.max_len = max_len
        self.do_lower_case = do_lower_case

        self.is_cuda = True if torch.cuda.is_available() else False
        self.device = torch.device("cuda:0" if self.is_cuda else "cpu")

        self.tokenizer = transformers.BertTokenizer.from_pretrained(self.pretrained_model_name_or_path, do_lower_case=self.do_lower_case)
        self.model = None

        self.id2label = None

    def _configure_model(self):
        """
        Configures model.

        Note that if model is pretrained from path from a
        previously trained BertModel, the parameters self.max_len
        and self.do_lower_case are retrieved from the configuration file.

        Returns
        -------
        model : transformers.modeling_bert.BertForSequenceClassification
            Bert model for sentence classification
        """

        model = self._get_model()
        if self.is_cuda:
            model.cuda()

        if constants.MODEL_INFO in model.config.to_dict():
            model_info = model.config.to_dict()[constants.MODEL_INFO]

            self.max_len = model_info[constants.PARAMS][constants.MAX_LEN]
            self.do_lower_case = model_info[constants.PARAMS][constants.DO_LOWER_CASE]

        return model

    def _get_model(self):
        """
        Utility function for self._configure_model(): returns
        pretrained Bert model for current task.

        Returns
        -------
        model : transformers.modeling_bert.*
            Bert model for current task

        """
        raise NotImplementedError("Class {} is child of class BertModel and must implement "
                                  "_get_model method!".format(self.__class__.__name__))

    def configure_dataset_loader(self, configs, path_to_config_folder, valid_size=0.2, is_return_datasets=False):
        """
        Loads data, performs preprocessing and configures dataset loader.

        Parameters
        ----------
        configs : dict
            Configuration dictionary
        path_to_config_folder : str
            Path to configuration file
        valid_size : float, default 0.2
            Dimension of validation dataset: it is used if no validation
            dataset is provided within the config file; validation daatset
            is obtained from training dataset with random splitting
        is_return_datasets : boolean, default False
            If True, datasets are returned

        Returns
        -------
        dataset_loader : dict
            Dictionary containing datasets loader
        (optional) datasets : dict
            Dictionary containing datasets

        """

        target_column = configs.get(constants.TARGET_COLUMN)

        # loading data
        datasets = models_utilities.load_datasets(configs, path_to_config_folder, valid_size)
        self.id2label = models_utilities.load_labels(os.path.join(path_to_config_folder, configs.get(constants.DATA_FOLDER)))
        if self.id2label is not None:
            log.info("Labels mapping: {}".format(self.id2label))

        # preprocessing data and generating dataset loader
        dataset_loader = {}
        for phase in constants.PHASES:
            if phase in datasets:
                self._preprocess_dataset(datasets[phase], configs)
                dataset_loader[phase] = self._get_dataset_loader(datasets[phase], phase, target_column)

                if phase == constants.TRAIN and self.max_len is not None:
                    log.info('Maximum sentence length: {}'.format(self.max_len))

                log.info("{} data dimension: {}".format(phase.title(), len(datasets[phase])))

        if is_return_datasets:
            return dataset_loader, datasets

        return dataset_loader

    def _preprocess_dataset(self, data, configs):
        """
        Auxiliary function for self._configure_dataset_loader():
        performs dataset preprocessing.

        Parameters
        ----------
        data : pandas.DataFrame
            Input data
        configs : dict
            Configuration dictionary

        """

        if self.is_single_sentence:
            sentence_column = configs.get(constants.SENTENCE_COLUMN)
            self.max_len = dataset_preprocessing.preprocess_dataset_single_sentence(data, sentence_column, self.max_len,
                                                                                    self.tokenizer, self.language)
        else:

            sentence1_column = configs.get(constants.FIRST_SENTENCE_COLUMN)
            sentence2_column = configs.get(constants.SECOND_SENTENCE_COLUMN)
            self.max_len = dataset_preprocessing.preprocess_dataset_sentences_pair(data, sentence1_column,
                                                                                   sentence2_column, self.max_len,
                                                                                   self.tokenizer, self.language)

    def _get_dataset_loader(self, data, phase, target_column):
        """
        Auxiliary function for self._configure_dataset_loader():
        returns dataset loader.

        Parameters
        ----------
        data : pandas.DataFrame
            Input data
        phase : str
            Phase
        target_column : str
            Column containing true labels

        Returns
        -------
        dataset_loader : dict
            Dictionary containing datasets loader

        """

        tensor_dataset = self._get_tensor_dataset(data, target_column)

        do_sampling = True if phase == constants.TRAIN else False
        sampler = torch.utils.data.RandomSampler(tensor_dataset) if do_sampling else torch.utils.data.SequentialSampler(
            tensor_dataset)
        dataset_loader = torch.utils.data.DataLoader(tensor_dataset, sampler=sampler, batch_size=self.batch_size)

        return dataset_loader

    def _get_tensor_dataset(self, data, target_column):
        """
        Auxiliary function for self._get_dataset_loader(): converts
        required values to torch.tensor and returns tensor dataset.

        Parameters
        ----------
        data : pandas.DataFrame
            Input data
        target_column : str
            Column containing true labels

        Returns
        -------
        torch.utils.data.dataset.TensorDataset
            Tensor dataset

        """
        raise NotImplementedError("Class {} is child of class BertModel and must implement "
                                  "_get_tensor_dataset method!".format(self.__class__.__name__))

    def train(self, dataset_loader, dataset_name='', metrics=[constants.METRICS.ACCURACY],
              evaluation_phases=[constants.VALID]):
        """
        Performs training of model on current training set,
        providing performances on training and other sets defined in
        evaluation_phases.

        Parameters
        ----------
        dataset_loader : dict
            Dictionary containing datasets loader
        dataset_name : str, default ''
            Name of current dataset
        metrics : list, default [constants.Metrics.accuracy]
            List of metrics computed on training and sets defined in
            evaluation_phases for each epoch; supported ones are
            defined in constants.Metrics
        evaluation_phases : list, default [constants.VALID]
            Evaluation phases performed, supported
            ones are defined in constants.phases

        Returns
        -------
        training_info : dict
            Training info, including learning curves computed during training,
            training time, training timestamp and metric evaluation.

        """

        # training configuration
        optimizer, scheduler = self._configure_training(len(dataset_loader[constants.TRAIN]))

        self._reset_randomness()
        phases = [constants.TRAIN] + evaluation_phases
        learning_curves = models_utilities.init_learning_curves(metrics, phases)
        start = time.time()
        results = {}

        for epoch in range(self.epochs):
            results = {}
            for phase in phases:
                start_phase = time.time()
                results[phase] = {metric : 0 for metric in metrics}
                results[phase][constants.METRICS.LOSS] = 0

                if phase == constants.TRAIN:
                    log.info('============ Epoch {} / {} ============'.format(epoch + 1, self.epochs))
                    log.info('Training...')
                    self.model.train()
                else:
                    log.info("{}...".format(phase.title()))
                    self.model.eval()

                n_batches = len(dataset_loader[phase])
                for i, batch in enumerate(dataset_loader[phase]):

                    if phase == constants.TRAIN and i % 50 == 0 and not i == 0:
                        log.info('\tBatch {}/{}\telapsed: {}.'.format(i, n_batches, self._format_time(time.time() - start_phase)))

                    # forward propagation
                    outputs, b_labels = self._batch_forward_propagation(batch, phase, is_add_labels=True)

                    # computing loss
                    loss = outputs[0]
                    logits = outputs[1]

                    results[phase][constants.METRICS.LOSS] += loss.item()
                    for metric in metrics:
                        results[phase][metric] += self._evaluate(logits, b_labels, metric)

                    if phase == constants.TRAIN:
                        # backward propagation
                        loss.backward()

                        # clip the norm of the gradients to 1.0: this is to help prevent the "exploding gradients" problem
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        optimizer.step()

                        # Update the learning rate.
                        scheduler.step()

                results[phase] = {m : value / n_batches for m, value in results[phase].items()}

                log.info("\tAverage {} loss: {:.2f}".format(phase, results[phase][constants.METRICS.LOSS]))
                for metric in metrics:
                    log.info("\tAverage {} {}: {:.2f}".format(phase, metric, results[phase][metric]))
                log.info("\tElapsed time {}: {}".format(phase, self._format_time(time.time() - start_phase)))

            models_utilities.update_learning_curves(learning_curves, results)

        log.info("Training complete!")
        training_info = {constants.DATASET_NAME : dataset_name,
                         constants.LEARNING_CURVES : learning_curves,
                         constants.RESULTS : {metric : results[constants.VALID][metric] for metric in metrics} if len(results) > 0 else {},
                         constants.TRAINED_AT : datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                         constants.TRAINING_TIME : (datetime.datetime.utcfromtimestamp(time.time() - start)).strftime('%H:%M:%S')}

        return training_info

    def _configure_training(self, n_batches_train):
        """
        Configures training component:
            1. optimizer
            2. scheduler

        Parameters
        ----------
        n_batches_train : int
            Number of batches of training data

        Returns
        -------
        optimizer : transformers.optimization.AdamW
            Optimizer
        scheduler : torch.optim.lr_scheduler.LambdaLR
            Scheduler

        """

        # Create optimizer
        params = filter(lambda x: x.requires_grad, self.model.parameters())
        optimizer = transformers.AdamW(params, lr=self.alpha, eps=1e-8)

        # Total number of training steps is number of batches * number of epochs.
        total_steps = n_batches_train * self.epochs

        # Create the learning rate scheduler
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                                 num_warmup_steps=0, # Default value in run_glue.py
                                                                 num_training_steps=total_steps)

        return optimizer, scheduler

    def _batch_forward_propagation(self, batch, phase, is_add_labels):
        """
        Performs forward propagation on given batch.

        Parameters
        ----------
        batch : list
            List containing components of input batch
        phase : str
            Phase
        is_add_labels : boolean
            If True, labels are given to the model in the
            forward propagation, otherwise are not.

        Returns
        -------
        outputs : tuple
            Outputs of the forward propagation. If is_add_labels is True,
            outputs is (loss, logits, ...), otherwise is (logits, ...).

        """
        raise NotImplementedError("Class {} is child of class BertModel and must implement "
                                  "_batch_forward_propagation method!".format(self.__class__.__name__))

    def _evaluate(self, logits, labels, metric):
        """
        Utility function: performs evaluation given logits
        and true labels.

        """
        raise NotImplementedError("Class {} is child of class BertModel and must implement "
                                  "_evaluate method!".format(self.__class__.__name__))

    def predict_dataset(self, dataset_loader, phase=constants.VALID):
        """
        Performs prediction on given dataset.

        Parameters
        ----------
        dataset_loader : dict
            Dictionary containing datasets loader
        phase : str, default constants.VALID
            Phase on which doing evaluation, supported
            ones are defined in constants.phases

        Returns
        -------
        y_pred : list
            Predicted labels

        """

        y_pred = []
        start = time.time()
        if phase not in dataset_loader:
            log.warning("Phase '{}' is not defined inside dataset loader. Available phases are: {}".format(phase, list(dataset_loader.keys())))
            return y_pred

        for batch in dataset_loader[phase]:
            outputs, _ = self._batch_forward_propagation(batch, phase, is_add_labels=False)

            logits = outputs[0]
            logits = logits.detach().cpu().numpy()

            y_pred += np.argmax(logits, axis=1).flatten().tolist()

        log.info("Elapsed time {} prediction: {}".format(phase, self._format_time(time.time() - start)))

        return y_pred

    def evaluate_dataset(self, dataset_loader, phase=constants.VALID, metrics=[constants.METRICS.ACCURACY]):
        """
        Performs evaluation on given dataset.

        Parameters
        ----------
        dataset_loader : dict
            Dictionary containing datasets loader
        phase : str, default constants.VALID
            Phase on which doing evaluation, supported
            ones are defined in constants.phases
        metrics : list, default [constants.Metrics.accuracy]
            List of metrics computed on given phase; supported ones are
            defined in constants.Metrics

        Returns
        -------
        results : dict
            Results

        """

        start = time.time()

        if phase not in dataset_loader:
            log.warning("Phase '{}' is not defined inside dataset loader. Available phases are: {}".format(phase, list(dataset_loader.keys())))
            return {}

        results = {metric : 0 for metric in metrics}
        for batch in dataset_loader[phase]:
            outputs, _ = self._batch_forward_propagation(batch, phase, is_add_labels=False)
            b_labels = batch[-1]

            logits = outputs[0]
            for metric in metrics:
                results[metric] += self._evaluate(logits, b_labels, metric)

        n_batches = len(dataset_loader[phase])
        results = {m: value / n_batches for m, value in results.items()}
        log.info("Elapsed time {} evaluation: {}".format(phase, self._format_time(time.time() - start)))

        return results

    def save_model(self, trained_models_folder, model_name, training_info):
        """
        Saves current model to file into folder
        {self.configs.get(constants.trained_models_folder)}/{model_name}.

        Parameters
        ----------
        trained_models_folder : str
            Folder for saving the model
        model_name : str
            Model name
        training_info : dict
            Training info, including learning curves computed during training,
            training time, training timestamp and metric evaluation.

        """

        model_folder = os.path.join(trained_models_folder, model_name)

        if os.path.exists(model_folder):
            log.warning("Input folder '{}' already exists! Cannot save model!".format(model_folder))
        else:

            os.mkdir(model_folder)
            # If we have a distributed model, save only the encapsulated model
            # (it was wrapped in PyTorch DistributedDataParallel or DataParallel)
            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model

            # If we save using the predefined names, we can load using `from_pretrained`
            output_model_file = os.path.join(model_folder, transformers.WEIGHTS_NAME)
            output_config_file = os.path.join(model_folder, transformers.CONFIG_NAME)

            torch.save(model_to_save.state_dict(), output_model_file)
            config_dict = model_to_save.config.to_dict()
            config_dict[constants.MODEL_INFO] = self._get_model_info(training_info)
            with open(output_config_file, 'w') as outfile:
                json.dump(config_dict, outfile)

            self.tokenizer.save_vocabulary(model_folder)
            log.info("Saved model in folder '{}'".format(model_folder))

            if self.id2label is not None:
                with open(os.path.join(model_folder, constants.LABELS_FILE), 'w') as outfile:
                    json.dump(self.id2label, outfile)

    def _get_model_info(self, training_info):
        """
        Auxiliary function for self.save_model(): returns
        dictionary of hyperparameters and configuration of
        current model, together with training information.

        Parameters
        ----------
        training_info : dict
            Training info, including learning curves computed during training,
            training time, training timestamp and metric evaluation.

        Returns
        -------
        model_info : dict
            Model information

        """

        params = {constants.ALPHA : self.alpha,
                  constants.BATCH_SIZE : self.batch_size,
                  constants.EPOCHS : self.epochs,
                  constants.RANDOM_STATE : self.random_state,
                  constants.MAX_LEN: self.max_len,
                  constants.DO_LOWER_CASE: self.do_lower_case}

        model_info = {constants.LANGUAGE : self.language,
                      constants.PARAMS : params,
                      constants.DATASET_NAME: training_info.get(constants.DATASET_NAME),
                      constants.PRETRAINED_MODEL_NAME: self.pretrained_model_name_or_path.split('/')[-1],
                      constants.TRAINED_AT : training_info.get(constants.TRAINED_AT),
                      constants.TRAINING_TIME : training_info.get(constants.TRAINING_TIME),
                      constants.RESULTS : training_info.get(constants.RESULTS)}

        return model_info

    def predict_example(self, *args):
        """
        Performs prediction on a new example depending on
        current task.

        """
        raise NotImplementedError("Class {} is child of class BertModel and must implement "
                                  "predict_example method!".format(self.__class__.__name__))

    def _reset_randomness(self):
        """
        Utility function: resets randomness.

        """

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True if self.is_cuda else False

        torch.manual_seed(self.random_state)
        random.seed(self.random_state)
        torch.manual_seed(self.random_state)

        if self.is_cuda:
            torch.cuda.manual_seed_all(self.random_state)

        np.random.seed(self.random_state)

    @staticmethod
    def _format_time(elapsed):
        """
        Utility function: takes a time in seconds and returns a string
        "hh:mm:ss".

        """

        elapsed_rounded = int(round(elapsed))

        return str(datetime.timedelta(seconds=elapsed_rounded))
