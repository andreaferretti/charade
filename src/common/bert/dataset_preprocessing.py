import transformers

import common.bert.albert_tokenizer as albert_tokenizer
import common.bert.constants as constants


def preprocess_dataset_single_sentence(data, sentence_column, max_len, tokenizer, language):
    """
    Performs preprocessing on input dataset in case of a task with
    single sentences. The text of the sentences is expected to be
    into the column {sentence_column}.
    The preprocessing performs the following steps:
        1. tokenization and mapping words to ids, then padding
        until {max_len};
        2. computation of attention mask vector (i.e. a list
        containing 1 if the corresponding token is not a padding,
        otherwise 0);
    then saves respectively in new columns constants.TOKEN_IDS
    and constants.ATT_MASK the computed lists.

    Note that if max_len is None, the maximum sentence length is computed
    as the maximum sentence length in given data.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data
    sentence_column : str
        Name of the column containing the sentence
    max_len : int or None, default None
        If not None, maximum length for sentences, otherwise the
        maximum length is taken from the training dataset
    tokenizer : transformers.tokenization_bert.BertTokenizer
        Tokenizer
    language : str
        Language, supported values are defined in constants.LANGUAGES

    Returns
    -------
    max_len : int
        Maximum length used for padding

    """

    data[sentence_column + '_TMP'] = data[sentence_column].copy()
    if language == constants.LANGUAGES.IT:
        albert_tokenizer_ = albert_tokenizer.AlBERTo_Preprocessing(do_lower_case=True)
        data[sentence_column + '_TMP'] = data[sentence_column].apply(lambda x: albert_tokenizer_.preprocess(x))

    # 1. tokenizing and mapping words to ids
    data[constants.TOKEN_IDS] = data[sentence_column + '_TMP'].apply(
        lambda x: tokenizer.encode('[CLS] {} [SEP]'.format(x), add_special_tokens=False))

    if max_len is None:
        max_len = min([data[constants.TOKEN_IDS].apply(lambda x: len(x)).max(), tokenizer.max_len])

    # 2. computing attention mask
    data[constants.ATT_MASK] = (data[constants.TOKEN_IDS]
                                .apply(lambda x: [1] * min([max_len, len(x)]) +
                                                 [0] * max([max_len - len(x), 0])))

    # 3. padding sentences to max_len
    data[constants.TOKEN_IDS] = (data[constants.TOKEN_IDS]
                                 .apply(lambda x: x[:max_len]
                                                  + [tokenizer.pad_token_id] * max([max_len - len(x), 0])))
    del data[sentence_column + '_TMP']

    return int(max_len)


def preprocess_dataset_sentences_pair(data, first_sentence_column, second_sentence_column, max_len, tokenizer, language):
    """
    Performs preprocessing on input dataset in case of a task with
    pairs of sentences. The text of the first sentences is expected to be
    into the column {first_sentence_column}, and the text of the second
    sentences into {second_sentence_column}.
    The preprocessing performs the following steps:
        1. tokenization and mapping words to ids, then padding
        until {max_len}, independently for each sentence (note that max_len
        refers to the maximum length of a single sentence);
        2. computation of token type ids vector related to the
        concatenation of the pair of sentences;
        3. computation of the concatenation of the token ids;
        4. computation of attention mask vector related to the concatenation
        of the pair of sentences;
    then saves in new columns constants.TOKEN_IDS, constants.ATT_MASK and
    constants.TOKEN_TYPE_IDS the computed lists.

    Note that if max_len is None, the maximum sentence length is computed as
    the maximum sentence length among first and second sentences in given data.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data
    first_sentence_column : str
        Name of the column containing the first sentence
    second_sentence_column : str
        Name of the column containing the second sentence
    max_len : int or None, default None
        If not None, maximum length for sentences, otherwise the
        maximum length is taken from the training dataset
    tokenizer : transformers.tokenization_bert.BertTokenizer
        Tokenizer
    language : str
        Language, supported values are defined in constants.LANGUAGES

    Returns
    -------
    max_len : int
        Maximum length used for padding

    """

    data[first_sentence_column + '_TMP'] = data[first_sentence_column].copy()
    data[second_sentence_column + '_TMP'] = data[second_sentence_column].copy()
    if language == constants.LANGUAGES.IT:
        albert_tokenizer_ = albert_tokenizer.AlBERTo_Preprocessing(do_lower_case=True)
        data[first_sentence_column + '_TMP'] = data[first_sentence_column].apply(lambda x: albert_tokenizer_.preprocess(x))
        data[second_sentence_column + '_TMP'] = data[second_sentence_column].apply(lambda x: albert_tokenizer_.preprocess(x))

    # 1. tokenizing and mapping words to ids for each sentence
    data[constants.TOKEN_IDS + '_1'] = data[first_sentence_column + '_TMP'].apply(lambda x: tokenizer.encode(x, add_special_tokens=False))
    data[constants.TOKEN_IDS + '_2'] = data[second_sentence_column + '_TMP'].apply(lambda x: tokenizer.encode(x, add_special_tokens=False))

    if max_len is None:
        max_len = data.apply(lambda x: max([len(x[constants.TOKEN_IDS + '_1']), len(x[constants.TOKEN_IDS + '_2'])]), axis = 1).max()
        max_len = min([max_len, int(tokenizer.max_len / 2)])

    # padding to max_len each sentence
    for suffix in ['_1', '_2']:
        data[constants.TOKEN_IDS + suffix] = (data[constants.TOKEN_IDS + suffix]
                                              .apply(lambda x: x[:max_len] + [tokenizer.pad_token_id] * max([max_len - len(x), 0])))


    # 2. computing token type ids of the concatenation of the sentences (i.e. 0 if a token belongs to first sentence, 1 otherwise)
    data[constants.TOKEN_TYPE_IDS] = data.apply(lambda x: [0] * (len(x[constants.TOKEN_IDS + '_1'][:max_len]) + 2) +
                                                          [1] * (len(x[constants.TOKEN_IDS + '_2'][:max_len]) + 1), axis = 1)

    # 3. computing the concatenation of token ids
    data[constants.TOKEN_IDS] = data.apply(lambda x: [tokenizer.cls_token_id] + x[constants.TOKEN_IDS + '_1'][:max_len] + [tokenizer.sep_token_id] +
                                                     x[constants.TOKEN_IDS + '_2'][:max_len] + [tokenizer.sep_token_id], axis = 1)

    # 4. computing attention mask
    data[constants.ATT_MASK] = data[constants.TOKEN_IDS].apply(lambda x: [1 if i != tokenizer.pad_token_id else 0 for i in x])

    del data[constants.TOKEN_IDS + '_1'], data[constants.TOKEN_IDS + '_2'], data[first_sentence_column + '_TMP'],  data[second_sentence_column + '_TMP']

    return int(max_len)

