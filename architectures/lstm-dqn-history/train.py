import os
import glob
import argparse
import pickle
import yaml

from tqdm import tqdm

import gym
import textworld.gym
from textworld import EnvInfos

from custom_agent import CustomAgent

# List of additional information available during evaluation.
AVAILABLE_INFORMATION = EnvInfos(
    description=True, inventory=True,
    max_score=True, objective=True, entities=True, verbs=True,
    command_templates=True, admissible_commands=True,
    has_won=True, has_lost=True,
    extras=["recipe"]
)


def _validate_requested_infos(infos: EnvInfos):
    msg = "The following information cannot be requested: {}"
    for key in infos.basics:
        if not getattr(AVAILABLE_INFORMATION, key):
            raise ValueError(msg.format(key))

    for key in infos.extras:
        if key not in AVAILABLE_INFORMATION.extras:
            raise ValueError(msg.format(key))


def train(game_files, config_file_name):

    agent = CustomAgent(config_file_name)
    requested_infos = agent.select_additional_infos()
    _validate_requested_infos(requested_infos)

    env_id = textworld.gym.register_games(game_files, requested_infos,
                                          max_episode_steps=agent.max_nb_steps_per_episode,
                                          name="training")
    env_id = textworld.gym.make_batch(env_id, batch_size=agent.batch_size, parallel=True)
    env = gym.make(env_id)
    config = {}
    with open(config_file_name) as reader:
        config = yaml.safe_load(reader)
    history_length = int(config['training']['nb_history'])
    full_stats = {}
    for epoch_no in range(1, agent.nb_epochs + 1):
        stats = {
            "scores": [],
            "steps": [],
        }
        for game_no in tqdm(range(len(game_files))):
            obs, infos = env.reset()
            agent.train()
            scores = [0] * len(obs)
            dones = [False] * len(obs)
            steps = [0] * len(obs)
            # makes a copy of initial instructions and puts them in an array with pre appended empty items
            empty_list = [""] * history_length
            history = list(map(lambda x : empty_list + [x], obs))
            history_step = 0 + history_length
            while not all(dones):
                # Increase step counts.
                steps = [step + int(not done) for step, done in zip(steps, dones)]
                history_step += 1
                history_obs =  list(map(lambda x: "".join(x[history_step-history_length:history_step]), history))
                commands = agent.act(history_obs, scores, dones, infos)
                obs, scores, dones, infos = env.step(commands)
                # append next step
                history = [x + [obs[i]] for i, x in enumerate(history)]
            # Let the agent knows the game is done.
            agent.act(obs, scores, dones, infos)

            stats["scores"].extend(scores)
            stats["steps"].extend(steps)
        full_stats[epoch_no] = stats

        score = sum(stats["scores"]) / agent.batch_size
        steps = sum(stats["steps"]) / agent.batch_size
        print("Epoch: {:3d} | {:2.1f} pts | {:4.1f} steps".format(epoch_no, score, steps))
    stats_file_name = config_file_name.split("/")
    stats_file_name[-2] = "stats"
    stats_file_name[-1] = stats_file_name[-1].strip("yaml") + "pickle"
    stats_file_name = "/".join(stats_file_name)
    with open(stats_file_name,"wb") as f:
        pickle.dump(full_stats, f)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train an agent.")
    parser.add_argument("games", metavar="game", nargs="+",
                       help="List of games (or folders containing games) to use for training.")
    parser.add_argument("-c", "--config", metavar="config", nargs="?", default="config.yaml", 
                       help="Single config.yaml file path should be provided")
    args = parser.parse_args()

    games = []
    for game in args.games:
        if os.path.isdir(game):
            games += glob.glob(os.path.join(game, "*.ulx"))
        else:
            games.append(game)

    print("{} games found for training.".format(len(games)))
    train(games, args.config)
