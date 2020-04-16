TOKEN_IDS = 'token_ids'
ATT_MASK = 'att_mask'
TOKEN_TYPE_IDS = 'token_type_ids'

# for single sentence tasks
SENTENCE_COLUMN = 'sentence_column'

# for pair of sentences tasks
FIRST_SENTENCE_COLUMN = 'first_sentence_column'
SECOND_SENTENCE_COLUMN = 'second_sentence_column'

TARGET_COLUMN = 'target_column'
DATA_FOLDER = 'data_folder'

CONFIG_FILE = 'configs.json'
LABELS_FILE = 'labels.json'


LEARNING_CURVES = 'learning_curves'


########## phases ##########
TRAIN = 'training'
TEST = 'test'
VALID = 'validation'

PHASES = [TRAIN, VALID, TEST]


class METRICS:
    ACCURACY = 'accuracy'
    PRECISION = 'precision'
    RECALL = 'recall'
    F1_SCORE = 'f1_score'
    LOSS = 'loss'


class LANGUAGES:
    IT = 'it'
    EN = 'en'


########## information for saving model ##########

MODEL_INFO = 'model-info'

IS_SINGLE_SENTENCE = 'is-single-sentence'
LANGUAGE = 'language'

# hyperparameters
PARAMS = 'params'
ALPHA = 'alpha'
BATCH_SIZE = 'batch-size'
EPOCHS = 'epochs'
RANDOM_STATE = 'random-state'
MAX_LEN = 'max-len'

# configuration
DATASET_NAME = 'dataset-name'
PRETRAINED_MODEL_NAME = 'pretrained-model-name'
DO_LOWER_CASE = 'do-lower-case'

# training info
TRAINED_AT = 'trained-at'
TRAINING_TIME = 'training-time'
RESULTS = 'results'