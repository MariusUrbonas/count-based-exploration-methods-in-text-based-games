import glob
from matplotlib import pyplot as plt
import numpy as np
import os
import pickle
import sys


NUM_EPOCHS = 400
NUM_GRAPHS = 4
STATS_FOLDERS = {
    'banana_baseline': ('DQN (Baseline)', '#1b9e77'),
    'obs-count-rep-motivation-false-v2': ('DQN-UCB-SA', '#d95f02'),
    'obs-count-rep-motivation-true-v2': ('DQN-MBIE-EB', '#7570b3'),
    'banana_cumulative': ('DQN-S+', '#e7298a'),
    'banana_episodic': ('DQN-S++', '#a6761d')
}


def plot_stats(axis, stats_folder, quest_length, label, color):
    stats_files = glob.glob('experiments/{}/stats/*ql-{}*.pickle'.format(stats_folder, quest_length))

    data_list = []
    for stats_file in stats_files:
        with open(stats_file, 'rb') as pickle_file:
            data = pickle.load(pickle_file)
            del(data['obs_set'])
            data_list.append([data[epoch]['steps'] for epoch in data])
    
    if len(data_list) == 0:
        return

    data_np = np.array(data_list)[:, :NUM_EPOCHS, :]
    data_mean = np.mean(data_np, (0, 2))
    data_std = np.std(data_np, (0, 2))

    axis.fill_between(
        np.arange(NUM_EPOCHS),
        data_mean - data_std,
        data_mean + data_std,
        color=color,
        alpha=0.05
    )
    axis.plot(
        np.arange(NUM_EPOCHS), 
        data_mean,
        color=color,
        label=label,
        linewidth=0.5
    )


# Get parameters
try:
    output_name = sys.argv[1]
except:
    print('Usage: pythonw make_graphs.py output_folder')
    exit()

fig, axes = plt.subplots(NUM_GRAPHS, 1, figsize=(5, 3 * NUM_GRAPHS), sharex=True, sharey=True)

# Make a separate subplot for each quest length
for axis, quest_length in zip(axes, range(1, NUM_GRAPHS + 1)):

    for stats_folder in STATS_FOLDERS:
        label, color = STATS_FOLDERS[stats_folder]
        plot_stats(axis, stats_folder, quest_length, label, color)
    # Set title and legend
    axis.set_title('Quest Length {}'.format(quest_length))
    if quest_length == 1:
        axis.legend()

    # Set axis limits
    axis.set_ylim(0, 100)
    axis.set_xlim(0, NUM_EPOCHS)

    # Set axis labels
    axis.set_ylabel('Steps')
    if quest_length == NUM_GRAPHS:
        axis.set_xlabel('Epoch')
    

plt.savefig('figures/{}.pdf'.format(output_name, quest_length), bbox_inches='tight')
