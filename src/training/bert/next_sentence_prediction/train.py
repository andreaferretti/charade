import argparse
import os
import sys



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
import common.bert.models.bert_for_next_sentence_prediction as bert_for_next_sentence_prediction
import common.bert.constants as constants



def parse_options():
    parser = argparse.ArgumentParser(description='Run Bert Model for Sentence Classification')
    parser.add_argument('--data-dir', required=True, help='the directory that contains the dataset')
    parser.add_argument('--data', required=True, help='the CSV file that contains the training sentences')
    parser.add_argument('--first-sentence-column', type=str, required=True, help='the column of the data containing first sentences')
    parser.add_argument('--second-sentence-column', type=str, required=True, help='the column of the data containing second sentences')
    parser.add_argument('--target-column', type=str, required=True, help='the column of the data containing targets')
    parser.add_argument('--lang', required=True, help='the language of the dataset')
    parser.add_argument('--valid-size', type=float, default=0.2, help='size of validation set')
    parser.add_argument('--num-epochs', type=int, default=4, help='number of epochs')
    parser.add_argument('--alpha', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--batch-size', type=int, default=10, help='batch size')
    parser.add_argument('--random-state', type=int, default=0, help='random state')
    parser.add_argument('--max-len', type=int, default=None, help='maximum sentence length')
    parser.add_argument('--do-lower-case', type=bool, default=True, help='do lower case')
    parser.add_argument('--pretrained-model-name-or-path', required=True, help='pretrained model name (e.g. "bert-base-uncased") or path')
    parser.add_argument('--model-name', required=True, help='the name of the model to generate')
    parser.add_argument('--model-dir', default=os.path.join('models', 'bert', 'classification'), help='the directory where to store the models')
    return parser.parse_args()



if __name__ == '__main__':

    options = parse_options()

    model = bert_for_next_sentence_prediction.BertForNextSentencePrediction(options.lang, options.pretrained_model_name_or_path,
                                                                            options.alpha, options.batch_size,
                                                                            options.num_epochs, options.random_state,
                                                                            options.max_len, options.do_lower_case)

    configs = {constants.DATA_FOLDER: os.path.join(options.data_dir),
               constants.TRAIN: options.data,
               constants.FIRST_SENTENCE_COLUMN: options.first_sentence_column,
               constants.SECOND_SENTENCE_COLUMN: options.second_sentence_column,
               constants.TARGET_COLUMN: options.target_column,
               constants.LANGUAGE: options.lang}

    path_to_config_folder = ''
    dataset_loader = model.configure_dataset_loader(configs, path_to_config_folder, options.valid_size)

    metrics = [constants.METRICS.ACCURACY, constants.METRICS.PRECISION, constants.METRICS.RECALL, constants.METRICS.F1_SCORE]
    evaluation_phases = [constants.VALID]

    training_info = model.train(dataset_loader, dataset_name=options.data.replace('.csv', ''), metrics=metrics, evaluation_phases=evaluation_phases)
    model.save_model(options.model_dir, options.model_name, training_info)