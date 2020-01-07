# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

# Compute log sum exp in a numerically stable way for the forward algorithm
def sum_in_log_space(matrix):
    maxes, _ = matrix.max(dim=1)
    diff = matrix - maxes.view(-1, 1)
    return maxes + torch.log(torch.exp(diff).sum(dim=1))


class Tagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tag_index, num_layers=1, bidirectional=True, dropout=0):
        super(Tagger, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.vocab_size = vocab_size
        self.tagset_size = len(tag_index)
        self.START = tag_index.index('<SOS>')
        self.END = tag_index.index('<EOS>')

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional, dropout=dropout)

        # Maps the output of the LSTM into tag space.
        hidden2tag_dim = 2 * hidden_dim if bidirectional else hidden_dim
        self.hidden2tag = nn.Linear(hidden2tag_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size).to(self.device))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[self.START, :] = -10000
        self.transitions.data[:, self.END] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        axis1_dim = 2 * self.lstm.num_layers if self.bidirectional else self.lstm.num_layers
        return (torch.randn(axis1_dim, 1, self.hidden_dim).to(self.device),
                torch.randn(axis1_dim, 1, self.hidden_dim).to(self.device))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        forward_var = torch.full((self.tagset_size,), -10000).to(self.device)
        # START_TAG has all of the score.
        forward_var[self.START] = 0

        # Iterate through the sentence
        for feat in feats:
            # alphas_t = []  # The forward tensors at this timestep
            # for next_tag in range(self.tagset_size):
            #     # broadcast the emission score: it is the same regardless of
            #     # the previous tag
            #     emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
            #     # the ith entry of trans_score is the score of transitioning to
            #     # next_tag from i
            #     trans_score = self.transitions[next_tag].view(1, -1)
            #     # The ith entry of next_tag_var is the value for the
            #     # edge (i -> next_tag) before we do log-sum-exp
            #     next_tag_var = forward_var + trans_score + emit_score
            #     # The forward variable for this tag is log-sum-exp of all the
            #     # scores.
            #     alphas_t.append(log_sum_exp(next_tag_var).view(1))
            next_tag_vars = self.transitions + feat.view(-1, 1) + forward_var
            # forward_var = torch.cat(alphas_t).view(1, -1)
            forward_var = sum_in_log_space(next_tag_vars)
        terminal_var = forward_var + self.transitions[self.END]
        alpha = sum_in_log_space(terminal_var.view(1, -1))
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), -1)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1).to(self.device)
        tags = torch.cat([torch.tensor([self.START], dtype=torch.long).to(self.device), tags])
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.END, tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []
        # feats is len(sentence) x num_tags

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.).to(self.device)
        init_vvars[0][self.START] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            # each feat is a vector of dimension num_tags
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.END]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.START  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, torch.tensor(tag_seq).to(self.device)