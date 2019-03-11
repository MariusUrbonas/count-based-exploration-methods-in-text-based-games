import logging
import numpy as np

import torch
import torch.nn.functional as F

from layers import Embedding, masked_mean, FastUniLSTM

logger = logging.getLogger(__name__)


class LSTM_DQN(torch.nn.Module):
    model_name = 'lstm_dqn'

    def __init__(self, model_config, word_vocab, generate_length=5, enable_cuda=False):
        super(LSTM_DQN, self).__init__()
        self.model_config = model_config
        self.enable_cuda = enable_cuda
        self.word_vocab_size = len(word_vocab)
        self.id2word = word_vocab
        self.generate_length = generate_length
        self.read_config()
        self._def_layers()
        self.init_weights()
        # self.print_parameters()

    def print_parameters(self):
        amount = 0
        for p in self.parameters():
            amount += np.prod(p.size())
        print("total number of parameters: %s" % (amount))
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        amount = 0
        for p in parameters:
            amount += np.prod(p.size())
        print("number of trainable parameters: %s" % (amount))

    def read_config(self):
        # model config
        self.embedding_size = self.model_config['embedding_size']
        self.encoder_rnn_hidden_size = self.model_config['encoder_rnn_hidden_size']
        self.action_scorer_hidden_dim = self.model_config['action_scorer_hidden_dim']
        self.dropout_between_rnn_layers = self.model_config['dropout_between_rnn_layers']

    def _def_layers(self):

        # word embeddings
        self.word_embedding = Embedding(embedding_size=self.embedding_size,
                                        vocab_size=self.word_vocab_size,
                                        enable_cuda=self.enable_cuda)

        # lstm encoder
        self.encoder = FastUniLSTM(ninp=self.embedding_size,
                                   nhids=self.encoder_rnn_hidden_size,
                                   dropout_between_rnn_layers=self.dropout_between_rnn_layers)

        self.action_scorer_shared = torch.nn.Linear(self.encoder_rnn_hidden_size[-1], self.action_scorer_hidden_dim)
        action_scorers = []
        for _ in range(self.generate_length):
            action_scorers.append(torch.nn.Linear(self.action_scorer_hidden_dim, self.word_vocab_size, bias=False))
        self.action_scorers = torch.nn.ModuleList(action_scorers)
        self.fake_recurrent_mask = None

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.action_scorer_shared.weight.data)
        for i in range(len(self.action_scorers)):
            torch.nn.init.xavier_uniform_(self.action_scorers[i].weight.data)
        self.action_scorer_shared.bias.data.fill_(0)

    def representation_generator(self, _input_words):
        embeddings, mask = self.word_embedding.forward(_input_words)  # batch x time x emb
        encoding_sequence, _, _ = self.encoder.forward(embeddings, mask)  # batch x time x h
        mean_encoding = masked_mean(encoding_sequence, mask)  # batch x h
        return mean_encoding

    def action_scorer(self, state_representation):
        hidden = self.action_scorer_shared.forward(state_representation)  # batch x hid
        hidden = F.relu(hidden)  # batch x hid
        action_ranks = []
        for i in range(len(self.action_scorers)):
            action_ranks.append(self.action_scorers[i].forward(hidden))  # batch x n_vocab
        return action_ranks
