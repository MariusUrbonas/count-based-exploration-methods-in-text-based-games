#!/usr/bin/env python3

import argparse
import docker
import glob
import gym
import json
import multiprocessing
import os
import requests
import subprocess
import sys
import tempfile
import textworld
import textworld.gym
import time
import tqdm


NB_EPISODES = 10
MAX_EPISODE_STEPS = 100
TIMEOUT = 6 * 3600  # 6 hours

# List of additional information available during evaluation.
AVAILABLE_INFORMATION = textworld.EnvInfos(
    max_score=True, has_won=True, has_lost=True,                    # Handicap 0
    description=True, inventory=True, objective=True,               # Handicap 1
    verbs=True, command_templates=True,                             # Handicap 2
    entities=True,                                                  # Handicap 3
    extras=["recipe"],                                              # Handicap 4
    admissible_commands=True,                                       # Handicap 5
)


def _validate_requested_infos(infos):
    msg = "The following information cannot be requested: {}"
    for key in infos.basics:
        if not getattr(AVAILABLE_INFORMATION, key):
            raise ValueError(msg.format(key))

    for key in infos.extras:
        if key not in AVAILABLE_INFORMATION.extras:
            raise ValueError(msg.format(key))


class _ReplayAgent:
    """
    An agent that replays the actions of another agent.
    """

    def __init__(self, stats):
        self._stats = stats
        self._game = None
        self._episode = 0
        self._step = 0

    def train(self):
        pass

    def eval(self):
        pass

    def select_additional_infos(self):
        infos = textworld.EnvInfos()
        for info in self._stats["requested_infos"]:
            if info in AVAILABLE_INFORMATION.extras:
                infos.extras.append(info)
            else:
                setattr(infos, info, True)
        return infos

    def act(self, obs, scores, dones, infos):
        if all(dones):
            self._episode += 1
            self._step = 0
            return

        if infos["_name"] != self._game:
            self._game = infos["_name"]
            self._episode = 0

        step = self._step
        self._step += 1

        command = self._stats["games"][self._game]["runs"][self._episode]["commands"][step]
        return [command]


def _play_game(agent_class, agent_class_args, gamefile):
    game_name = os.path.basename(gamefile)

    if agent_class_args:
        agent = agent_class(agent_class_args)
    else:
        agent = agent_class()

    agent.eval()
    requested_infos = agent.select_additional_infos()
    _validate_requested_infos(requested_infos)

    # Turn on flags needed for the evaluation.
    requested_infos.has_won = True
    requested_infos.has_lost = True
    requested_infos.max_score = True

    stats = {}
    start_time = time.time()

    stats["runs"] = []

    name = "test_{}".format(hash(gamefile))
    env_id = textworld.gym.register_games([gamefile], requested_infos,
                                            max_episode_steps=MAX_EPISODE_STEPS,
                                            name=name)
    env_id = textworld.gym.make_batch(env_id, batch_size=1)
    env = gym.make(env_id)

    for no_episode in range(NB_EPISODES):
        obs, infos = env.reset()

        all_commands = []
        scores = [0] * len(obs)
        dones = [False] * len(obs)
        steps = [0] * len(obs)
        while not all(dones):
            # Increase step counts.
            steps = [step + int(not done) for step, done in zip(steps, dones)]

            # HACK to get the replay agent the current game
            if isinstance(agent, _ReplayAgent):
                infos["_name"] = game_name

            commands = agent.act(obs, scores, dones, infos)
            all_commands.append(commands)
            obs, scores, dones, infos = env.step(commands)

        # Let the agent knows the game is done.
        agent.act(obs, scores, dones, infos)

        # Collect stats
        stats["runs"].append({})
        stats["runs"][no_episode]["score"] = scores[0]
        stats["runs"][no_episode]["steps"] = steps[0]
        stats["runs"][no_episode]["commands"] = [cmds[0] for cmds in all_commands]
        stats["runs"][no_episode]["has_won"] = infos["has_won"][0]
        stats["runs"][no_episode]["has_lost"] = infos["has_lost"][0]

    env.close()
    stats["max_scores"] = infos["max_score"][0]
    elapsed = time.time() - start_time
    stats["duration"] = elapsed

    return {game_name: stats}, requested_infos.basics + requested_infos.extras


def evaluate(agent_class, agent_class_args, game_files, nb_processes):
    stats = {"games": {}, "requested_infos": [], "game_files": game_files}

    print("Using {} processes.".format(nb_processes))
    desc = "Evaluating {} games".format(len(game_files))
    pbar = tqdm.tqdm(total=len(game_files), desc=desc)

    def _assemble_results(args):
        data, requested_infos = args
        stats["games"].update(data)
        stats["requested_infos"] = requested_infos

        game_name, infos = list(data.items())[0]
        total_scores = sum(d["score"] for d in infos["runs"])
        total_steps = sum(d["steps"] for d in infos["runs"])

        desc = "{:2d} / {}:\t{}".format(total_scores, total_steps, game_name)
        pbar.write(desc)
        pbar.update()

    if nb_processes > 1:
        pool = multiprocessing.Pool(nb_processes)
        results = []
        for game_file in game_files:
            result = pool.apply_async(_play_game, (agent_class, agent_class_args, game_file), callback=_assemble_results)
            results.append(result)

        for result in results:
            result.get()

        pool.close()
        pool.join()
        pbar.close()

    else:
        for game_file in game_files:
            data = _play_game(agent_class, agent_class_args, game_file)
            _assemble_results(data)

        pbar.close()

    return stats


def _run_evaluation(agent_class, args, agent_class_args=None):
    games = glob.glob(os.path.join(args.games_dir, "**/*.ulx"), recursive=True)
    stats = evaluate(agent_class, agent_class_args, games, args.nb_processes)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    with open(args.output, "w") as f:
        json.dump(stats, f)


def _dockerize(args):
    submission_dir = os.path.abspath(args.submission_dir)
    games_dir = os.path.abspath(args.games_dir)
    output_dir = os.path.dirname(os.path.abspath(args.output))
    self_file = os.path.abspath(__file__)

    with tempfile.NamedTemporaryFile(dir=output_dir) as output_file:
        client = docker.from_env()

        image_path = os.path.join(submission_dir, "Dockerimage")
        if os.path.exists(image_path):
            with open(image_path, "r") as f:
                image = f.read().strip()
        else:
            image = "tavianator/textworld-codalab"

        volumes = {
            submission_dir: {
                "bind": "/usr/src/submission",
                "mode": "ro",
            },
            games_dir: {
                "bind": "/usr/share/textworld-games",
                "mode": "ro",
            },
            output_file.name: {
                "bind": "/usr/share/textworld-stats.json",
                "mode": "rw",
            },
            self_file: {
                "bind": "/usr/bin/evaluate.py",
                "mode": "ro",
            },
        }

        command = [
            "python3",
            "/usr/bin/evaluate.py",
            "--in-docker",
            "/usr/src/submission",
            "/usr/share/textworld-games",
            "/usr/share/textworld-stats.json",
        ]

        if args.debug:
            command += ["--debug"]

        print("Loading {}...".format(image))
        container = client.containers.run(
            image,
            command,
            detach=True,
            network_mode="none",
            volumes=volumes,
            environment=["PYTHONUNBUFFERED=1", "MKL_NUM_THREADS=1", "OMP_NUM_THREADS=1"],
        )

        try:
            print("Running {}...".format(image))
            result = container.wait(timeout=TIMEOUT)
        finally:
            if args.debug:
                sys.stdout.buffer.write(container.logs(stdout=True, stderr=False))
                sys.stderr.buffer.write(container.logs(stdout=False, stderr=True))

            container.remove(force=True)

        if result["StatusCode"] != 0:
            msg = ("Some errors occur when evaluating the agent. You can test your agent"
                   " using the `test_submission.py` script provided with the starting kit"
                   " and using the `--debug` flag."
                   " If you can't find your error, reach out to us: textworld@microsoft.com.")
            raise NameError(msg)

        print("Done")
        stats = json.load(output_file)

    _run_evaluation(_ReplayAgent, args, agent_class_args=stats)


def main():
    parser = argparse.ArgumentParser(description="Evaluate an agent.")
    parser.add_argument("--in-docker", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument("submission_dir")
    parser.add_argument("games_dir")
    parser.add_argument("output", nargs='?', default="stats.json")
    parser.add_argument("--nb-processes", type=int)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    args.nb_processes = args.nb_processes or multiprocessing.cpu_count()
    if args.debug:
        args.nb_processes = 1

    if args.in_docker:
        args.submission_dir = os.path.abspath(args.submission_dir)
        args.games_dir = os.path.abspath(args.games_dir)
        args.output = os.path.abspath(args.output)
        os.chdir(args.submission_dir)  # Needed to load local files (e.g. vocab.txt)
        sys.path = [args.submission_dir] + sys.path  # Prepend to PYTHONPATH
        from custom_agent import CustomAgent
        _run_evaluation(CustomAgent, args)
    else:
        _dockerize(args)

if __name__ == "__main__":
    main()
