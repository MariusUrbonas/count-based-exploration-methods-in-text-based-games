import glob
from matplotlib import pyplot as plt
import numpy as np
import os
import pickle
import sys


NUM_EPOCHS = 500
STATS_FOLDERS = {
    'baseline_smaller': ('Baseline', '#ef8a62'),
    'counting_episodic_smaller': ('Counting (episodic)', '#67a9cf')
}


def plot_stats(stats_folder, quest_length, label, color):
    stats_files = glob.glob('experiments/{}/stats/*ql-{}*.pickle'.format(stats_folder, quest_length))

    data_list = []
    for stats_file in stats_files:
        with open(stats_file, 'rb') as pickle_file:
            data = pickle.load(pickle_file)
            data_list.append([data[epoch]['steps'] for epoch in data])
    
    if len(data_list) == 0:
        return

    data_np = np.array(data_list)[:, :NUM_EPOCHS, :]
    data_mean = np.mean(data_np, (0, 2))
    data_std = np.std(data_np, (0, 2))

    plt.fill_between(
        np.arange(NUM_EPOCHS),
        data_mean - data_std,
        data_mean + data_std,
        color=color,
        alpha=0.15
    )
    plt.plot(
        np.arange(NUM_EPOCHS), 
        data_mean,
        color=color,
        label=label
    )


# Get parameters
try:
    output_folder = sys.argv[1]
except:
    print('Usage: pythonw make_graphs.py output_folder')
    exit()

# Make a separate graph for each quest length
for i in range(5):
    quest_length = i + 1

    plt.figure()
    plt.ylim(0, 100)
    plt.xlim(0, NUM_EPOCHS)
    plt.title('Quest Length {}'.format(quest_length))
    
    for stats_folder in STATS_FOLDERS:
        label, color = STATS_FOLDERS[stats_folder]
        plot_stats(stats_folder, quest_length, label, color)

    # plt.show()
    plt.savefig('figures/{}/quest-length-{}.pdf'.format(output_folder, quest_length), bbox_inches='tight')
