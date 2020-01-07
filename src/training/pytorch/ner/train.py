import random
import argparse
import sys
import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from common.pytorch.ner.model import Tagger


def lines(path):
    with open(path) as f:
        return [l[:-1] for l in f.readlines()]

def invert(xs):
    return { t: i for i, t in enumerate(xs) }

def harmonic_mean(a, b):
    if a == 0 or b == 0:
        return 0
    m = ((1 / a) + (1 / b)) / 2
    return 1 / m

def print_stat(name, value):
    print('%s: %.2f%%' % (name, (100 * value)))

def run_epoch(model, criterion, optimizer, data, eos, sos_tag):
    words = data['words']
    tags = data['tags']
    sos_offset = 1 if sos_tag == None else 2

    print('Training...')

    count, epoch_loss = 0, 0
    for i, j in zip(eos, eos[1:]):
        print('%s/%s' % (count, len(eos)-1), end='\r')
        count += 1

        # <EOS>, <SOS>, ..., <EOS>, <SOS>, ...
        sentence = words[i+sos_offset:j]
        sentence_tags = tags[i+sos_offset:j]

        optimizer.zero_grad()
        loss = criterion(sentence, sentence_tags)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    print('Epoch avg loss: %.6f' % (epoch_loss / count))


def compute_stats(model, data, eos, nop_tag, sos_tag):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # removes initial <SOS> tag if present
    sos_offset = 1 if sos_tag == None else 2

    words = data['words']
    tags = data['tags']

    print('Computing accuracy...')

    count = 0
    correct = 0
    nop_predicted_as_nop = 0
    nop_predicted_as_tag = 0
    tag_predicted_correctly = 0
    tag_predicted_as_nop = 0
    tag_predicted_as_other_tag = 0
    for i, j in zip(eos, eos[1:]):
        print('%s/%s' % (count, len(eos)-1), end='\r')
        count += 1

        sentence = words[i+sos_offset:j]
        real_tags = tags[i+sos_offset:j]

        model.zero_grad()
        _, predicted_tags = model(sentence)
        predicted_tags = torch.tensor(predicted_tags).to(device)
        real_tags_nop = real_tags == nop_tag
        predicted_tags_nop = predicted_tags == nop_tag
        matches = real_tags == predicted_tags
        nop_predicted_as_nop += (real_tags_nop * matches).sum().item()
        nop_predicted_as_tag += (real_tags_nop * (1 - matches)).sum().item()
        tag_predicted_correctly += ((1 - real_tags_nop) * matches).sum().item()
        tag_predicted_as_nop += ((1 - real_tags_nop) * (1 - matches) * predicted_tags_nop).sum().item()
        tag_predicted_as_other_tag += ((1 - real_tags_nop) * (1 - matches) * (1 - predicted_tags_nop)).sum().item()

    #print(tag_predicted_correctly, nop_predicted_as_tag, nop_predicted_as_nop, tag_predicted_as_other_tag, tag_predicted_as_nop)

    predicted_as_tag = tag_predicted_correctly + nop_predicted_as_tag + tag_predicted_as_other_tag
    actual_tags = tag_predicted_correctly + tag_predicted_as_nop + tag_predicted_as_other_tag

    precision = tag_predicted_correctly / predicted_as_tag if (predicted_as_tag > 0) else 0
    recall = tag_predicted_correctly / actual_tags if (actual_tags > 0) else 0
    f1 = harmonic_mean(precision, recall)

    #SOS and EOS are not tags to be predicted
    tags_to_predict = tag_predicted_correctly + tag_predicted_as_nop + nop_predicted_as_tag + nop_predicted_as_nop + tag_predicted_as_other_tag
    accuracy = (nop_predicted_as_nop + tag_predicted_correctly) / tags_to_predict

    print_stat('Accuracy', accuracy)
    print_stat('Precision', precision)
    print_stat('Recall', recall)
    print_stat('F1-score', f1)
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


def show_example(model, data, eos, indices, sos_tag):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    words = data['words']
    tags = data['tags']
    word_index = indices['words']
    tag_index = indices['tags']

    i = random.randint(0, len(eos)-2)
    sos_offset = 1 if sos_tag==None else 2
    start, end = eos[i], eos[i+1]

    sentence = words[start+sos_offset:end]
    real_tags = tags[start+sos_offset:end]

    text = ' '.join(word_index[i] for i in sentence.data)
    real_tag_text = ' '.join(tag_index[i] for i in real_tags.data)
    print('> ' + text)
    print('Actual tags:')
    print('> ' + real_tag_text)

    _, predicted_tags = model(sentence)
    predicted_tags = torch.tensor(predicted_tags).to(device)
    predicted_tags_text = ' '.join(tag_index[i] for i in predicted_tags)
    print('Predicted tags:')
    print('> ' + predicted_tags_text)


def write_results(stats, options, epoch):
    if options.results is not None:
        results = {
            'epoch': epoch,
            'params': {
                'num-epochs': options.num_epochs,
                'model': options.model,
                'train-words': options.train_words,
                'train-tags': options.train_tags,
                'test-words': options.test_words,
                'test-tags': options.test_tags,
                'embedding': options.embeddings,
                'learning-rate': options.learning_rate,
                'momentum': options.momentum,
                'dropout': options.dropout,
                'num-layers': options.num_layers,
                'hidden-dim': options.hidden_dim,
                'bidirectional': not options.unidirectional
            },
            'metrics': stats
        }

        with open(options.results, 'w') as f:
            json.dump(results, f)


def parse_options():
    parser = argparse.ArgumentParser(description='Run LSTM')
    parser.add_argument('--train-words', required=True, help='the file that contains the tensor with the training inputs')
    parser.add_argument('--train-tags', required=True, help='the file that contains the tensor with the training labels')
    parser.add_argument('--test-words', required=True, help='the file that contains the tensor with the test inputs')
    parser.add_argument('--test-tags', required=True, help='the file that contains the tensor with the test labels')
    parser.add_argument('--eos-limit', type=int, default=None, help='number of sentences to use for train and test. Tipically used during debug to reduce epoch time.')
    parser.add_argument('--word-index', required=True, help='the file that contains the word index')
    parser.add_argument('--tag-index', required=True, help='the file that contains the tag index')
    parser.add_argument('--model', required=True, help='the model file')
    parser.add_argument('--results', help='the file where the performances of the saved model will be written')
    parser.add_argument('--embeddings', help='optional word embeddings')
    parser.add_argument('--num-epochs', type=int, default=30, help='number of training epochs')
    parser.add_argument('--num-layers', type=int, default=1, help='number of RNN layers')
    parser.add_argument('--hidden-dim', type=int, default=300, help='number of neurons of each RNN hidden layer')
    parser.add_argument('--unidirectional', action='store_true', default=False, help='if this option is given, unidirectional (not bidirectiona) RNN is created')
    parser.add_argument('--learning-rate', type=float, default=0.1, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.8, help='momentum')
    parser.add_argument('--dropout', default=0, type=float, help='dropout')
    parser.add_argument('--resume', action='store_true', default=False, help='if True model is loaded from model path, else a new model is created')
    return parser.parse_args()


def main():
    torch.manual_seed(1)
    options = parse_options()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_words = torch.load(options.train_words).to(device)
    train_tags = torch.load(options.train_tags).to(device)
    test_words = torch.load(options.test_words).to(device)
    test_tags = torch.load(options.test_tags).to(device)
    word_index = lines(options.word_index)
    tag_index = lines(options.tag_index)

    if options.embeddings is not None:
        embeddings = torch.load(options.embeddings).to(device)
        embedding_len, embedding_dim = embeddings.shape
        if embedding_len!=len(word_index):
            raise Exception("number of words vectors in embedding %d != number of words in index %s" %(embedding_len, len(word_index)))
    else:
        embeddings = None
        embedding_dim = 300

    sos_tag = tag_index.index('<SOS>') if '<SOS>' in tag_index else None
    eos_tag = tag_index.index('<EOS>')
    nop_tag = tag_index.index('O')

    train_eos = (train_tags == eos_tag).nonzero().squeeze().tolist()
    test_eos = (test_tags == eos_tag).nonzero().squeeze().tolist()
    train_eos = train_eos if options.eos_limit==None else train_eos[:options.eos_limit]
    test_eos = test_eos if options.eos_limit==None else test_eos[:options.eos_limit]

    print('Number of training sentences: %s' % (len(train_eos) - 1))
    print('Number of test sentences: %s' % (len(test_eos) - 1))

    if options.resume:
        with open(options.model, 'rb') as f:
            model = torch.load(f)
        print('model resumed')
    else:
        model = Tagger(
            vocab_size=len(word_index),
            tag_index=tag_index,
            embedding_dim=embedding_dim,
            hidden_dim=options.hidden_dim,
            num_layers=options.num_layers,
            dropout=options.dropout,
            bidirectional=not options.unidirectional
        )
    model = model.to(device)

    criterion = model.neg_log_likelihood
    optimizer = optim.SGD(model.parameters(), lr=options.learning_rate, momentum=options.momentum)

    train_data = {
        'words': train_words,
        'tags': train_tags
    }
    test_data = {
        'words': test_words,
        'tags': test_tags
    }
    indices = {
        'words': word_index,
        'tags': tag_index
    }

    best_f1 = 0
    for epoch in range(options.num_epochs):
        print('====Epoch %s of %s====' % (epoch + 1, options.num_epochs))
        run_epoch(model, criterion, optimizer, train_data, train_eos, sos_tag)
        show_example(model, train_data, train_eos, indices, sos_tag)
        stats = compute_stats(model, test_data, test_eos, nop_tag, sos_tag)
        f1 = stats['f1']
        if f1 > best_f1:
            best_f1 = f1
            with open(options.model, 'wb') as f:
                torch.save(model, options.model)
                write_results(stats, options, epoch)

if __name__ == '__main__':
    main()
