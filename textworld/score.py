import os
import sys
import json
import shutil
import argparse
from os.path import join as pjoin

import numpy as np


MAX_HANDICAP = 5
HANDICAP_ADJUSTMENTS = {
    0: 1.00,
    1: 0.85,
    2: 0.77,
    3: 0.73,
    4: 0.65,
    5: 0.50,
}


def get_total_score(stats):
    score = 0
    for gamefile in stats:
        for no_episode in range(len(stats[gamefile]["runs"])):
            score += stats[gamefile]["runs"][no_episode]["score"]

    return score


def get_total_steps(stats):
    steps = 0
    for gamefile in stats:
        for no_episode in range(len(stats[gamefile]["runs"])):
            steps += stats[gamefile]["runs"][no_episode]["steps"]

    return steps


def get_handicap(requested_infos):
    requested_infos = set(requested_infos)
    handicap = 0

    if len(requested_infos & {"description", "inventory"}) > 0:
        handicap = 1

    if len(requested_infos & {"verbs", "command_templates"}) > 0:
        handicap = 2

    if len(requested_infos & {"entities"}) > 0:
        handicap = 3

    if len(requested_infos & {"recipe"}) > 0:
        handicap = 4

    if len(requested_infos & {"admissible_commands"}) > 0:
        handicap = 5

    return handicap


def score_leaderboard(stats, output_dir):
    # Get agent's handicap.
    handicap = get_handicap(stats["requested_infos"])

    # Extract result from stats.
    leaderboard = {}
    leaderboard["score"] = get_total_score(stats["games"])
    leaderboard["adjusted_score"] = HANDICAP_ADJUSTMENTS[handicap] * leaderboard["score"]
    leaderboard["nb_steps"] = get_total_steps(stats["games"])
    leaderboard["handicap"] = get_handicap(stats["requested_infos"])

    # Write leaderboard results.
    if not os.path.exists(output_dir):
	    os.makedirs(output_dir)

    content = "\n".join("{}: {}".format(k, v) for k, v in leaderboard.items())
    with open(pjoin(output_dir, "scores.txt"), "w") as f:
        f.write(content)

    print(content)


def score_html(stats, output_dir):
    html = "Available during the validation phase."

    # Write detailed results.
    html_dir = pjoin(output_dir, "html")
    if not os.path.exists(html_dir):
	    os.makedirs(html_dir)

    with open(pjoin(html_dir, "detailed_results.html"), "w") as f:
        f.write(html)

def main():
    parser = argparse.ArgumentParser(description="Extract score from `stats.json`.")
    parser.add_argument("stats", help="JSON file")
    parser.add_argument("output_dir")
    args = parser.parse_args()

    with open(args.stats) as f:
        stats = json.load(f)

    score_leaderboard(stats, args.output_dir)
    score_html(stats, args.output_dir)


if __name__ == "__main__":
    main()
