import subprocess

# Experiments will be generated in `tw_games/experiment_name`
experiment_name = 'game_size_exploration'

# These stay constant
world_size = 3
num_objects = 7

# These change
quest_lengths = [1, 2, 3, 4, 5]
seeds = [1234]  # Add more seeds to generate more versions of each game type

for quest_length in quest_lengths:
    for seed in seeds:
        game_name = 'ws-{}_ql-{}_no-{}_seed-{}'.format(
            world_size,
            quest_length,
            num_objects,
            seed
        )

        subprocess.run([
            'tw-make', 'custom',
            '--world-size', str(world_size),
            '--nb-objects', str(num_objects), 
            '--quest-length', str(quest_length),
            '--seed', str(seed),
            '--output', 'tw_games/{}/{}.ulx'.format(experiment_name, game_name)
        ])
