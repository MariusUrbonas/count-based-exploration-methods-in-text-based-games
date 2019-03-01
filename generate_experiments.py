from string import Template
import subprocess
import sys
import os

# Games will be generated in `tw_games/experiment_name`
# Config files will be generated in `config_files/experiment_name`
try:
    experiment_name = sys.argv[1]
except IndexError:
    print('Usage: `python generate_experiments your_experiment_name`')
    exit()

try:
    os.mkdir('config_files/{}'.format(experiment_name))
except Exception:
    print('Directory `config_files/{}` already exists'.format(experiment_name))

# These stay constant
world_size = 3
num_objects = 7

# These change
quest_lengths = [1, 2, 3, 4, 5]
seeds = [1234]  # Add more seeds to generate more versions of each game type

for quest_length in quest_lengths:
    for seed in seeds:
        
        # Create game name, for use in all file names
        game_name = 'ws-{}_ql-{}_no-{}_seed-{}'.format(
            world_size,
            quest_length,
            num_objects,
            seed
        )
        if False:
            # Generate textworld game with specified params
            subprocess.run([
                'tw-make', 'custom',
                '--world-size', str(world_size),
                '--nb-objects', str(num_objects), 
                '--quest-length', str(quest_length),
                '--seed', str(seed),
                '--output', 'tw_games/{}/{}.ulx'.format(experiment_name, game_name)
            ])

        # Generate YAML config file
        with open('base_config.yaml') as base_config:
            template = Template(base_config.read())
            config_file_name = 'config_files/{}/{}.yaml'.format(experiment_name, game_name)

            with open(config_file_name, 'w') as config_file:
                config_file.write(template.substitute({
                    'experiment_name': experiment_name,
                    'game_name': game_name
                }))
