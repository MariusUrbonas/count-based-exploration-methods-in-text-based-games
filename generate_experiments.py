from string import Template
import subprocess
import sys
import os

# Games will be generated in `experiments/experiment_name/games`
# Config files will be generated in `experiments/experiment_name/configs`
# Train scripts will be generated in `experiments/experiment_name/scripts`
try:
    experiment_name = sys.argv[1]
except IndexError:
    print('Usage: `python generate_experiments your_experiment_name`')
    exit()

try:
    os.mkdir('experiments/' + experiment_name)
    os.mkdir('experiments/{}/games'.format(experiment_name))
    os.mkdir('experiments/{}/config'.format(experiment_name))
    os.mkdir('experiments/{}/scripts'.format(experiment_name))
    os.mkdir('experiments/{}/models'.format(experiment_name))
except Exception:
    print('Directory `experiments/{}` already exists.'.format(experiment_name))
    exit()

# Parameters for the generated games
world_size = 3
num_objects = 7
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

        print('\nGenerating experiments for {}...\n'.format(game_name))

        # Generate textworld game with specified params
        subprocess.run([
            'tw-make', 'custom',
            '--world-size', str(world_size),
            '--nb-objects', str(num_objects), 
            '--quest-length', str(quest_length),
            '--seed', str(seed),
            '--output', 'experiments/{}/games/{}.ulx'.format(experiment_name, game_name)
        ])

        print('Generated game')

        # Generate YAML config file
        config_file_name = 'experiments/{}/config/{}.yaml'.format(experiment_name, game_name)
        with open('base_config.yaml') as base_config:
            template = Template(base_config.read())
            with open(config_file_name, 'w') as config_file:
                config_file.write(template.substitute({
                    'experiment_name': experiment_name,
                    'game_name': game_name
                }))
        
        print('Generated config file')

        # Generate training scripts
        script_file_name = 'experiments/{}/scripts/{}.sh'.format(experiment_name, game_name)
        with open('base_script.sh') as base_script:
            template = Template(base_script.read())
            with open(script_file_name, 'w') as script_file:
                script_file.write(template.substitute({
                    'experiment_name': experiment_name,
                    'game_name': game_name,
                    'config_file_name': config_file_name
                }))

        print('Generated training scripts')

        os.makedirs('experimemts/{}/models/{}'.format(experiment_name, game_name))

        print('Created folder to save models')
