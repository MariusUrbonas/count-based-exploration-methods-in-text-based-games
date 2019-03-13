import glob
from matplotlib import pyplot as plt
import numpy as np
import os
import pickle
import sys


# Get parameters
try:
    stats_folder_baseline = sys.argv[1]
    stats_folder_comparison = sys.argv[2]
    output_folder = sys.argv[3]
except:
    print('Usage: pythonw make_graphs.py stats_folder_baseline stats_folder_comparison output_folder')
    exit()

# Make output folder
try:
    os.mkdir('figures/{}'.format(output_folder))
except:
    print('Output folder already exists')
    exit()

num_epochs = 501

# Make a separate graph for each quest length
for i in range(5):
    quest_length = i + 1

    plt.figure()
    plt.ylim(0, 100)
    plt.xlim(0, 500)

    # Get stats files for different seeds
    stats_files_comparison = glob.glob('{}/*ql-{}*.pickle'.format(stats_folder_comparison, quest_length))

    comparison_data = []
    for stats_file in stats_files_comparison:
        with open(stats_file, 'rb') as pickle_file:
            data = pickle.load(pickle_file)
            comparison_data.append([data[epoch]['steps'] for epoch in data])
    comparison_data_np = np.array(comparison_data)

    comparison_data_mean = np.mean(comparison_data_np, (0, 2))
    comparison_data_std = np.std(comparison_data_np, (0, 2))

    plt.fill_between(
        np.arange(num_epochs),
        comparison_data_mean - comparison_data_std,
        comparison_data_mean + comparison_data_std,
        color='#c4daff'
    )
    plt.plot(
        np.arange(num_epochs), 
        comparison_data_mean,
        color='#216eef'
    )
    
    # plt.show()
    plt.savefig('figures/{}/quest-length-{}.pdf'.format(output_folder, quest_length), bbox_inches='tight')
