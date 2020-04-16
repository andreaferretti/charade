import pandas as pd
import random
import os
import sys

import nltk

nltk.download('gutenberg')
nltk.download('punkt')



SENTENCE1 = 'sentence1'
SENTENCE2 = 'sentence2'

ID1 = 'id1'
ID2 = 'id2'

RANDOM_ID = 'random_id'
IS_NEXT = 'is_next'



def generate_dataset(gutenberg_corpus = 'carroll-alice.txt', dataset = 'alice_NSP', basedir=os.path.join('data', 'next_sentence_prediction')):

    sentences = [' '.join(s) for s in nltk.corpus.gutenberg.sents(gutenberg_corpus)]

    data = pd.DataFrame({SENTENCE1 : sentences, ID1 : list(range(len(sentences)))})

    # sentence next to 'CHAPTER X' sentence is the title of the chapter --> remove both sentences
    chapter_ids = data[data[SENTENCE1].apply(lambda x: 'CHAPTER' in x)][ID1].tolist()
    end_chapter_ids = [i - 1 for i in chapter_ids]
    chapter_ids += [i + 1 for i in chapter_ids]

    data['IS_CHAPTER'] = data[ID1].apply(lambda x: x in chapter_ids)
    data['IS_END_CHAPTER'] = data[ID1].apply(lambda x: x in end_chapter_ids)

    # selecting ids of sentences feasible as random next sentence
    choice_ids = [i for i in range(len(sentences)) if i not in chapter_ids]

    # selecting random ids
    random_ids = random.choices(choice_ids, k=len(sentences))
    data[RANDOM_ID] = random_ids

    # selecting next sentence: with probability 0.5 is the actual next sentence,
    # with probability 0.5 is the selected random sentence
    data[IS_NEXT] = [1 if random.random() < 0.5 else 0 for i in range(len(sentences) - 1)] + [0]

    data[ID2] = data.apply(lambda x: x[ID1] + 1 if x[IS_NEXT] == 1 else x[RANDOM_ID], axis=1)
    data[SENTENCE2] = data[ID2].apply(lambda x: sentences[x])

    # removing sentences 'CHAPTER X' and last sentences of a chapter
    filtered_data = data[data.apply(lambda x: True not in [x['IS_CHAPTER'], x['IS_END_CHAPTER']], axis=1)][[SENTENCE1, SENTENCE2, ID1, ID2, IS_NEXT]]
    filtered_data.to_csv(os.path.join(basedir, '{}.csv'.format(dataset)), index=False)
    print("Generated dataset of dimension: {}".format(len(filtered_data)))



if __name__ == '__main__':
    generate_dataset(gutenberg_corpus=sys.argv[1], dataset=sys.argv[2], basedir=sys.argv[3])