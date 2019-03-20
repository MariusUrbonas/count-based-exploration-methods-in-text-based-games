import glob
import numpy as np
import pickle


NUM_EPOCHS = 400
NUM_QLS = 4
STATS_FOLDERS = {  # 'folder': ('label', '#hexcolor')
    'banana_baseline': 'DQN (Baseline)',
    'dqn-ubc-sa': 'DQN-UCB-SA',
    'dqn-mbie-eb': 'DQN-MBIE-EB',
    'banana_cumulative': 'DQN-S+',
    'banana_episodic': 'DQN-S++',
    'cool_agent': 'DQN-KM++',
}

for stats_folder in STATS_FOLDERS:
    print(STATS_FOLDERS[stats_folder], end=' ')

    for quest_length in range(1, NUM_QLS + 1):
        stats_files = glob.glob('experiments/{}/stats/*ql-{}*.pickle'.format(stats_folder, quest_length))

        data_list = []
        for stats_file in stats_files:
            with open(stats_file, 'rb') as pickle_file:
                data = pickle.load(pickle_file)
                if 'obs_set' in data:
                    del(data['obs_set'])
                data_list.append([data[epoch]['steps'] for epoch in data][:NUM_EPOCHS])

        data_np = np.array(data_list)      
        print('& ${:2.1f}$ & ${:2.1f}$'.format(data_np.mean(), np.std(data_np, (0, 2)).mean()), end=' ')

    print('\\\\')
