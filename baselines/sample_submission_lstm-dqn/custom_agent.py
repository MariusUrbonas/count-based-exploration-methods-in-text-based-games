import os
import random
import yaml
import copy
from typing import List, Dict, Any
from collections import namedtuple

import spacy
import numpy as np

import torch
import torch.nn.functional as F

from textworld import EnvInfos

from model import LSTM_DQN
from generic import to_np, to_pt, preproc, _words_to_ids, pad_sequences, max_len


# a snapshot of state to be stored in replay memory
Transition = namedtuple('Transition', ('observation_id_list', 'word_indices',
                                       'reward', 'mask', 'done',
                                       'next_observation_id_list',
                                       'next_word_masks'))


class HistoryScoreCache(object):

    def __init__(self, capacity=1):
        self.capacity = capacity
        self.reset()

    def push(self, stuff):
        """stuff is float."""
        if len(self.memory) < self.capacity:
            self.memory.append(stuff)
        else:
            self.memory = self.memory[1:] + [stuff]

    def get_avg(self):
        return np.mean(np.array(self.memory))

    def reset(self):
        self.memory = []

    def __len__(self):
        return len(self.memory)


class PrioritizedReplayMemory(object):

    def __init__(self, capacity=100000, priority_fraction=0.0):
        # prioritized replay memory
        self.priority_fraction = priority_fraction
        self.alpha_capacity = int(capacity * priority_fraction)
        self.beta_capacity = capacity - self.alpha_capacity
        self.alpha_memory, self.beta_memory = [], []
        self.alpha_position, self.beta_position = 0, 0

    def push(self, is_prior=False, *args):
        """Saves a transition."""
        if self.priority_fraction == 0.0:
            is_prior = False
        if is_prior:
            if len(self.alpha_memory) < self.alpha_capacity:
                self.alpha_memory.append(None)
            self.alpha_memory[self.alpha_position] = Transition(*args)
            self.alpha_position = (self.alpha_position + 1) % self.alpha_capacity
        else:
            if len(self.beta_memory) < self.beta_capacity:
                self.beta_memory.append(None)
            self.beta_memory[self.beta_position] = Transition(*args)
            self.beta_position = (self.beta_position + 1) % self.beta_capacity

    def sample(self, batch_size):
        if self.priority_fraction == 0.0:
            from_beta = min(batch_size, len(self.beta_memory))
            res = random.sample(self.beta_memory, from_beta)
        else:
            from_alpha = min(int(self.priority_fraction * batch_size), len(self.alpha_memory))
            from_beta = min(batch_size - int(self.priority_fraction * batch_size), len(self.beta_memory))
            res = random.sample(self.alpha_memory, from_alpha) + random.sample(self.beta_memory, from_beta)
        random.shuffle(res)
        return res

    def __len__(self):
        return len(self.alpha_memory) + len(self.beta_memory)


class CustomAgent:
    def __init__(self):
        """
        Arguments:
            word_vocab: List of words supported.
        """
        self.mode = "train"
        with open("./vocab.txt") as f:
            self.word_vocab = f.read().split("\n")
        with open("config.yaml") as reader:
            self.config = yaml.safe_load(reader)
        self.word2id = {}
        for i, w in enumerate(self.word_vocab):
            self.word2id[w] = i
        self.EOS_id = self.word2id["</S>"]

        self.batch_size = self.config['training']['batch_size']
        self.max_nb_steps_per_episode = self.config['training']['max_nb_steps_per_episode']
        self.nb_epochs = self.config['training']['nb_epochs']

        # Set the random seed manually for reproducibility.
        np.random.seed(self.config['general']['random_seed'])
        torch.manual_seed(self.config['general']['random_seed'])
        if torch.cuda.is_available():
            if not self.config['general']['use_cuda']:
                print("WARNING: CUDA device detected but 'use_cuda: false' found in config.yaml")
                self.use_cuda = False
            else:
                torch.backends.cudnn.deterministic = True
                torch.cuda.manual_seed(self.config['general']['random_seed'])
                self.use_cuda = True
        else:
            self.use_cuda = False

        self.model = LSTM_DQN(model_config=self.config["model"],
                              word_vocab=self.word_vocab,
                              enable_cuda=self.use_cuda)

        self.experiment_tag = self.config['checkpoint']['experiment_tag']
        self.model_checkpoint_path = self.config['checkpoint']['model_checkpoint_path']
        self.save_frequency = self.config['checkpoint']['save_frequency']
        if not os.path.exists(self.model_checkpoint_path):
            os.mkdir(self.model_checkpoint_path)

        if self.config['checkpoint']['load_pretrained']:
            self.load_pretrained_model(self.model_checkpoint_path + '/' + self.config['checkpoint']['pretrained_experiment_tag'] + '.pt')
        if self.use_cuda:
            self.model.cuda()

        self.replay_batch_size = self.config['general']['replay_batch_size']
        self.replay_memory = PrioritizedReplayMemory(self.config['general']['replay_memory_capacity'],
                                                     priority_fraction=self.config['general']['replay_memory_priority_fraction'])

        # optimizer
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=self.config['training']['optimizer']['learning_rate'])

        # epsilon greedy
        self.epsilon_anneal_episodes = self.config['general']['epsilon_anneal_episodes']
        self.epsilon_anneal_from = self.config['general']['epsilon_anneal_from']
        self.epsilon_anneal_to = self.config['general']['epsilon_anneal_to']
        self.epsilon = self.epsilon_anneal_from
        self.update_per_k_game_steps = self.config['general']['update_per_k_game_steps']
        self.clip_grad_norm = self.config['training']['optimizer']['clip_grad_norm']

        self.nlp = spacy.load('en', disable=['ner', 'parser', 'tagger'])
        self.preposition_map = {"take": "from",
                                "chop": "with",
                                "slice": "with",
                                "dice": "with",
                                "cook": "with",
                                "insert": "into",
                                "put": "on"}
        self.single_word_verbs = set(["inventory", "look"])
        self.discount_gamma = self.config['general']['discount_gamma']
        self.current_episode = 0
        self.current_step = 0
        self._epsiode_has_started = False
        self.history_avg_scores = HistoryScoreCache(capacity=1000)
        self.best_avg_score_so_far = 0.0

    def train(self):
        """
        Tell the agent that it's training phase.
        """
        self.mode = "train"
        self.model.train()

    def eval(self):
        """
        Tell the agent that it's evaluation phase.
        """
        self.mode = "eval"
        self.model.eval()

    def _start_episode(self, obs: List[str], infos: Dict[str, List[Any]]) -> None:
        """
        Prepare the agent for the upcoming episode.

        Arguments:
            obs: Initial feedback for each game.
            infos: Additional information for each game.
        """
        self.init(obs, infos)
        self._epsiode_has_started = True

    def _end_episode(self, obs: List[str], scores: List[int], infos: Dict[str, List[Any]]) -> None:
        """
        Tell the agent the episode has terminated.

        Arguments:
            obs: Previous command's feedback for each game.
            score: The score obtained so far for each game.
            infos: Additional information for each game.
        """
        self.finish()
        self._epsiode_has_started = False

    def load_pretrained_model(self, load_from):
        """
        Load pretrained checkpoint from file.

        Arguments:
            load_from: File name of the pretrained model checkpoint.
        """
        print("loading model from %s\n" % (load_from))
        try:
            if self.use_cuda:
                state_dict = torch.load(load_from)
            else:
                state_dict = torch.load(load_from, map_location='cpu')
            self.model.load_state_dict(state_dict)
        except:
            print("Failed to load checkpoint...")

    def select_additional_infos(self) -> EnvInfos:
        """
        Returns what additional information should be made available at each game step.

        Requested information will be included within the `infos` dictionary
        passed to `CustomAgent.act()`. To request specific information, create a
        :py:class:`textworld.EnvInfos <textworld.envs.wrappers.filter.EnvInfos>`
        and set the appropriate attributes to `True`. The possible choices are:

        * `description`: text description of the current room, i.e. output of the `look` command;
        * `inventory`: text listing of the player's inventory, i.e. output of the `inventory` command;
        * `max_score`: maximum reachable score of the game;
        * `objective`: objective of the game described in text;
        * `entities`: names of all entities in the game;
        * `verbs`: verbs understood by the the game;
        * `command_templates`: templates for commands understood by the the game;
        * `admissible_commands`: all commands relevant to the current state;

        In addition to the standard information, game specific information
        can be requested by appending corresponding strings to the `extras`
        attribute. For this competition, the possible extras are:

        * `'recipe'`: description of the cookbook;
        * `'walkthrough'`: one possible solution to the game (not guaranteed to be optimal);

        Example:
            Here is an example of how to request information and retrieve it.

            >>> from textworld import EnvInfos
            >>> request_infos = EnvInfos(description=True, inventory=True, extras=["recipe"])
            ...
            >>> env = gym.make(env_id)
            >>> ob, infos = env.reset()
            >>> print(infos["description"])
            >>> print(infos["inventory"])
            >>> print(infos["extra.recipe"])

        Notes:
            The following information *won't* be available at test time:

            * 'walkthrough'
        """
        request_infos = EnvInfos()
        request_infos.description = True
        request_infos.inventory = True
        request_infos.entities = True
        request_infos.verbs = True
        request_infos.extras = ["recipe"]
        return request_infos

    def init(self, obs: List[str], infos: Dict[str, List[Any]]):
        """
        Prepare the agent for the upcoming games.

        Arguments:
            obs: Previous command's feedback for each game.
            infos: Additional information for each game.
        """
        # reset agent, get vocabulary masks for verbs / adjectives / nouns
        self.scores = []
        self.dones = []
        self.prev_actions = ["" for _ in range(len(obs))]
        # get word masks
        batch_size = len(infos["verbs"])
        verbs_word_list = infos["verbs"]
        noun_word_list, adj_word_list = [], []
        for entities in infos["entities"]:
            tmp_nouns, tmp_adjs = [], []
            for name in entities:
                split = name.split()
                tmp_nouns.append(split[-1])
                if len(split) > 1:
                    tmp_adjs += split[:-1]
            noun_word_list.append(list(set(tmp_nouns)))
            adj_word_list.append(list(set(tmp_adjs)))

        verb_mask = np.zeros((batch_size, len(self.word_vocab)), dtype="float32")
        noun_mask = np.zeros((batch_size, len(self.word_vocab)), dtype="float32")
        adj_mask = np.zeros((batch_size, len(self.word_vocab)), dtype="float32")
        for i in range(batch_size):
            for w in verbs_word_list[i]:
                if w in self.word2id:
                    verb_mask[i][self.word2id[w]] = 1.0
            for w in noun_word_list[i]:
                if w in self.word2id:
                    noun_mask[i][self.word2id[w]] = 1.0
            for w in adj_word_list[i]:
                if w in self.word2id:
                    adj_mask[i][self.word2id[w]] = 1.0
        second_noun_mask = copy.copy(noun_mask)
        second_adj_mask = copy.copy(adj_mask)
        second_noun_mask[:, self.EOS_id] = 1.0
        adj_mask[:, self.EOS_id] = 1.0
        second_adj_mask[:, self.EOS_id] = 1.0
        self.word_masks_np = [verb_mask, adj_mask, noun_mask, second_adj_mask, second_noun_mask]

        self.cache_description_id_list = None
        self.cache_chosen_indices = None
        self.current_step = 0

    def get_game_step_info(self, obs: List[str], infos: Dict[str, List[Any]]):
        """
        Get all the available information, and concat them together to be tensor for
        a neural model. we use post padding here, all information are tokenized here.

        Arguments:
            obs: Previous command's feedback for each game.
            infos: Additional information for each game.
        """
        inventory_token_list = [preproc(item, tokenizer=self.nlp) for item in infos["inventory"]]
        inventory_id_list = [_words_to_ids(tokens, self.word2id) for tokens in inventory_token_list]

        feedback_token_list = [preproc(item, str_type='feedback', tokenizer=self.nlp) for item in obs]
        feedback_id_list = [_words_to_ids(tokens, self.word2id) for tokens in feedback_token_list]

        quest_token_list = [preproc(item, tokenizer=self.nlp) for item in infos["extra.recipe"]]
        quest_id_list = [_words_to_ids(tokens, self.word2id) for tokens in quest_token_list]

        prev_action_token_list = [preproc(item, tokenizer=self.nlp) for item in self.prev_actions]
        prev_action_id_list = [_words_to_ids(tokens, self.word2id) for tokens in prev_action_token_list]

        description_token_list = [preproc(item, tokenizer=self.nlp) for item in infos["description"]]
        for i, d in enumerate(description_token_list):
            if len(d) == 0:
                description_token_list[i] = ["end"]  # if empty description, insert word "end"
        description_id_list = [_words_to_ids(tokens, self.word2id) for tokens in description_token_list]
        description_id_list = [_d + _i + _q + _f + _pa for (_d, _i, _q, _f, _pa) in zip(description_id_list, inventory_id_list, quest_id_list, feedback_id_list, prev_action_id_list)]

        input_description = pad_sequences(description_id_list, maxlen=max_len(description_id_list)).astype('int32')
        input_description = to_pt(input_description, self.use_cuda)

        return input_description, description_id_list

    def word_ids_to_commands(self, verb, adj, noun, adj_2, noun_2):
        """
        Turn the 5 indices into actual command strings.

        Arguments:
            verb: Index of the guessing verb in vocabulary
            adj: Index of the guessing adjective in vocabulary
            noun: Index of the guessing noun in vocabulary
            adj_2: Index of the second guessing adjective in vocabulary
            noun_2: Index of the second guessing noun in vocabulary
        """
        # turns 5 indices into actual command strings
        if self.word_vocab[verb] in self.single_word_verbs:
            return self.word_vocab[verb]
        if adj == self.EOS_id:
            res = self.word_vocab[verb] + " " + self.word_vocab[noun]
        else:
            res = self.word_vocab[verb] + " " + self.word_vocab[adj] + " " + self.word_vocab[noun]
        if self.word_vocab[verb] not in self.preposition_map:
            return res
        if noun_2 == self.EOS_id:
            return res
        prep = self.preposition_map[self.word_vocab[verb]]
        if adj_2 == self.EOS_id:
            res = res + " " + prep + " " + self.word_vocab[noun_2]
        else:
            res =  res + " " + prep + " " + self.word_vocab[adj_2] + " " + self.word_vocab[noun_2]
        return res

    def get_chosen_strings(self, chosen_indices):
        """
        Turns list of word indices into actual command strings.

        Arguments:
            chosen_indices: Word indices chosen by model.
        """
        chosen_indices_np = [to_np(item)[:, 0] for item in chosen_indices]
        res_str = []
        batch_size = chosen_indices_np[0].shape[0]
        for i in range(batch_size):
            verb, adj, noun, adj_2, noun_2 = chosen_indices_np[0][i],\
                                             chosen_indices_np[1][i],\
                                             chosen_indices_np[2][i],\
                                             chosen_indices_np[3][i],\
                                             chosen_indices_np[4][i]
            res_str.append(self.word_ids_to_commands(verb, adj, noun, adj_2, noun_2))
        return res_str

    def choose_random_command(self, word_ranks, word_masks_np):
        """
        Generate a command randomly, for epsilon greedy.

        Arguments:
            word_ranks: Q values for each word by model.action_scorer.
            word_masks_np: Vocabulary masks for words depending on their type (verb, adj, noun).
        """
        batch_size = word_ranks[0].size(0)
        word_ranks_np = [to_np(item) for item in word_ranks]  # list of batch x n_vocab
        word_ranks_np = [r * m for r, m in zip(word_ranks_np, word_masks_np)]  # list of batch x n_vocab
        word_indices = []
        for i in range(len(word_ranks_np)):
            indices = []
            for j in range(batch_size):
                msk = word_masks_np[i][j]  # vocab
                indices.append(np.random.choice(len(msk), p=msk / np.sum(msk, -1)))
            word_indices.append(np.array(indices))
        # word_indices: list of batch
        word_qvalues = [[] for _ in word_masks_np]
        for i in range(batch_size):
            for j in range(len(word_qvalues)):
                word_qvalues[j].append(word_ranks[j][i][word_indices[j][i]])
        word_qvalues = [torch.stack(item) for item in word_qvalues]
        word_indices = [to_pt(item, self.use_cuda) for item in word_indices]
        word_indices = [item.unsqueeze(-1) for item in word_indices]  # list of batch x 1
        return word_qvalues, word_indices

    def choose_maxQ_command(self, word_ranks, word_masks_np):
        """
        Generate a command by maximum q values, for epsilon greedy.

        Arguments:
            word_ranks: Q values for each word by model.action_scorer.
            word_masks_np: Vocabulary masks for words depending on their type (verb, adj, noun).
        """
        batch_size = word_ranks[0].size(0)
        word_ranks_np = [to_np(item) for item in word_ranks]  # list of batch x n_vocab
        word_ranks_np = [r - np.min(r) for r in word_ranks_np] # minus the min value, so that all values are non-negative
        word_ranks_np = [r * m for r, m in zip(word_ranks_np, word_masks_np)]  # list of batch x n_vocab
        word_indices = [np.argmax(item, -1) for item in word_ranks_np]  # list of batch
        word_qvalues = [[] for _ in word_masks_np]
        for i in range(batch_size):
            for j in range(len(word_qvalues)):
                word_qvalues[j].append(word_ranks[j][i][word_indices[j][i]])
        word_qvalues = [torch.stack(item) for item in word_qvalues]
        word_indices = [to_pt(item, self.use_cuda) for item in word_indices]
        word_indices = [item.unsqueeze(-1) for item in word_indices]  # list of batch x 1
        return word_qvalues, word_indices

    def get_ranks(self, input_description):
        """
        Given input description tensor, call model forward, to get Q values of words.

        Arguments:
            input_description: Input tensors, which include all the information chosen in
            select_additional_infos() concatenated together.
        """
        state_representation = self.model.representation_generator(input_description)
        word_ranks = self.model.action_scorer(state_representation)  # each element in list has batch x n_vocab size
        return word_ranks

    def act_eval(self, obs: List[str], scores: List[int], dones: List[bool], infos: Dict[str, List[Any]]) -> List[str]:
        """
        Acts upon the current list of observations, during evaluation.

        One text command must be returned for each observation.

        Arguments:
            obs: Previous command's feedback for each game.
            score: The score obtained so far for each game (at previous step).
            done: Whether a game is finished (at previous step).
            infos: Additional information for each game.

        Returns:
            Text commands to be performed (one per observation).

        Notes:
            Commands returned for games marked as `done` have no effect.
            The states for finished games are simply copy over until all
            games are done, in which case `CustomAgent.finish()` is called
            instead.
        """

        if self.current_step > 0:
            # append scores / dones from previous step into memory
            self.scores.append(scores)
            self.dones.append(dones)

        if all(dones):
            self._end_episode(obs, scores, infos)
            return  # Nothing to return.

        input_description, _ = self.get_game_step_info(obs, infos)
        word_ranks = self.get_ranks(input_description)  # list of batch x vocab
        _, word_indices_maxq = self.choose_maxQ_command(word_ranks, self.word_masks_np)

        chosen_indices = word_indices_maxq
        chosen_indices = [item.detach() for item in chosen_indices]
        chosen_strings = self.get_chosen_strings(chosen_indices)
        self.prev_actions = chosen_strings
        self.current_step += 1

        return chosen_strings

    def act(self, obs: List[str], scores: List[int], dones: List[bool], infos: Dict[str, List[Any]]) -> List[str]:
        """
        Acts upon the current list of observations.

        One text command must be returned for each observation.

        Arguments:
            obs: Previous command's feedback for each game.
            score: The score obtained so far for each game (at previous step).
            done: Whether a game is finished (at previous step).
            infos: Additional information for each game.

        Returns:
            Text commands to be performed (one per observation).

        Notes:
            Commands returned for games marked as `done` have no effect.
            The states for finished games are simply copy over until all
            games are done, in which case `CustomAgent.finish()` is called
            instead.
        """
        if not self._epsiode_has_started:
            self._start_episode(obs, infos)

        if self.mode == "eval":
            return self.act_eval(obs, scores, dones, infos)

        if self.current_step > 0:
            # append scores / dones from previous step into memory
            self.scores.append(scores)
            self.dones.append(dones)
            # compute previous step's rewards and masks
            rewards_np, rewards, mask_np, mask = self.compute_reward()

        input_description, description_id_list = self.get_game_step_info(obs, infos)
        # generate commands for one game step, epsilon greedy is applied, i.e.,
        # there is epsilon of chance to generate random commands
        word_ranks = self.get_ranks(input_description)  # list of batch x vocab
        _, word_indices_maxq = self.choose_maxQ_command(word_ranks, self.word_masks_np)
        _, word_indices_random = self.choose_random_command(word_ranks, self.word_masks_np)
        # random number for epsilon greedy
        rand_num = np.random.uniform(low=0.0, high=1.0, size=(input_description.size(0), 1))
        less_than_epsilon = (rand_num < self.epsilon).astype("float32")  # batch
        greater_than_epsilon = 1.0 - less_than_epsilon
        less_than_epsilon = to_pt(less_than_epsilon, self.use_cuda, type='float')
        greater_than_epsilon = to_pt(greater_than_epsilon, self.use_cuda, type='float')
        less_than_epsilon, greater_than_epsilon = less_than_epsilon.long(), greater_than_epsilon.long()

        chosen_indices = [less_than_epsilon * idx_random + greater_than_epsilon * idx_maxq for idx_random, idx_maxq in zip(word_indices_random, word_indices_maxq)]
        chosen_indices = [item.detach() for item in chosen_indices]
        chosen_strings = self.get_chosen_strings(chosen_indices)
        self.prev_actions = chosen_strings

        # push info from previous game step into replay memory
        if self.current_step > 0:
            for b in range(len(obs)):
                if mask_np[b] == 0:
                    continue
                is_prior = rewards_np[b] > 0.0
                self.replay_memory.push(is_prior, self.cache_description_id_list[b], [item[b] for item in self.cache_chosen_indices], rewards[b], mask[b], dones[b], description_id_list[b], [item[b] for item in self.word_masks_np])

        # cache new info in current game step into caches
        self.cache_description_id_list = description_id_list
        self.cache_chosen_indices = chosen_indices

        # update neural model by replaying snapshots in replay memory
        if self.current_step > 0 and self.current_step % self.update_per_k_game_steps == 0:
            loss = self.update()
            if loss is not None:
                # Backpropagate
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.optimizer.step()  # apply gradients

        self.current_step += 1

        if all(dones):
            self._end_episode(obs, scores, infos)
            return  # Nothing to return.
        return chosen_strings

    def compute_reward(self):
        """
        Compute rewards by agent. Note this is different from what the training/evaluation
        scripts do. Agent keeps track of scores and other game information for training purpose.

        """
        # mask = 1 if game is not finished or just finished at current step
        if len(self.dones) == 1:
            # it's not possible to finish a game at 0th step
            mask = [1.0 for _ in self.dones[-1]]
        else:
            assert len(self.dones) > 1
            mask = [1.0 if not self.dones[-2][i] else 0.0 for i in range(len(self.dones[-1]))]
        mask = np.array(mask, dtype='float32')
        mask_pt = to_pt(mask, self.use_cuda, type='float')
        # rewards returned by game engine are always accumulated value the
        # agent have recieved. so the reward it gets in the current game step
        # is the new value minus values at previous step.
        rewards = np.array(self.scores[-1], dtype='float32')  # batch
        if len(self.scores) > 1:
            prev_rewards = np.array(self.scores[-2], dtype='float32')
            rewards = rewards - prev_rewards
        rewards_pt = to_pt(rewards, self.use_cuda, type='float')

        return rewards, rewards_pt, mask, mask_pt

    def update(self):
        """
        Update neural model in agent. In this example we follow algorithm
        of updating model in dqn with replay memory.

        """
        if len(self.replay_memory) < self.replay_batch_size:
            return None
        transitions = self.replay_memory.sample(self.replay_batch_size)
        batch = Transition(*zip(*transitions))

        observation_id_list = pad_sequences(batch.observation_id_list, maxlen=max_len(batch.observation_id_list)).astype('int32')
        input_observation = to_pt(observation_id_list, self.use_cuda)
        next_observation_id_list = pad_sequences(batch.next_observation_id_list, maxlen=max_len(batch.next_observation_id_list)).astype('int32')
        next_input_observation = to_pt(next_observation_id_list, self.use_cuda)
        chosen_indices = list(list(zip(*batch.word_indices)))
        chosen_indices = [torch.stack(item, 0) for item in chosen_indices]  # list of batch x 1

        word_ranks = self.get_ranks(input_observation)  # list of batch x vocab
        word_qvalues = [w_rank.gather(1, idx).squeeze(-1) for w_rank, idx in zip(word_ranks, chosen_indices)]  # list of batch
        q_value = torch.mean(torch.stack(word_qvalues, -1), -1)  # batch

        next_word_ranks = self.get_ranks(next_input_observation)  # batch x n_verb, batch x n_noun, batchx n_second_noun
        next_word_masks = list(list(zip(*batch.next_word_masks)))
        next_word_masks = [np.stack(item, 0) for item in next_word_masks]
        next_word_qvalues, _ = self.choose_maxQ_command(next_word_ranks, next_word_masks)
        next_q_value = torch.mean(torch.stack(next_word_qvalues, -1), -1)  # batch
        next_q_value = next_q_value.detach()

        rewards = torch.stack(batch.reward)  # batch
        not_done = 1.0 - np.array(batch.done, dtype='float32')  # batch
        not_done = to_pt(not_done, self.use_cuda, type='float')
        rewards = rewards + not_done * next_q_value * self.discount_gamma  # batch
        mask = torch.stack(batch.mask)  # batch
        loss = F.smooth_l1_loss(q_value * mask, rewards * mask)
        return loss

    def finish(self) -> None:
        """
        All games in the batch are finished. One can choose to save checkpoints,
        evaluate on validation set, or do parameter annealing here.

        """
        # Game has finished (either win, lose, or exhausted all the given steps).
        self.final_rewards = np.array(self.scores[-1], dtype='float32')  # batch
        dones = []
        for d in self.dones:
            d = np.array([float(dd) for dd in d], dtype='float32')
            dones.append(d)
        dones = np.array(dones)
        step_used = 1.0 - dones
        self.step_used_before_done = np.sum(step_used, 0)  # batch

        self.history_avg_scores.push(np.mean(self.final_rewards))
        # save checkpoint
        if self.mode == "train" and self.current_episode % self.save_frequency == 0:
            avg_score = self.history_avg_scores.get_avg()
            if avg_score > self.best_avg_score_so_far:
                self.best_avg_score_so_far = avg_score

                save_to = self.model_checkpoint_path + '/' + self.experiment_tag + "_episode_" + str(self.current_episode) + ".pt"
                torch.save(self.model.state_dict(), save_to)
                print("========= saved checkpoint =========")

        self.current_episode += 1
        # annealing
        if self.current_episode < self.epsilon_anneal_episodes:
            self.epsilon -= (self.epsilon_anneal_from - self.epsilon_anneal_to) / float(self.epsilon_anneal_episodes)
