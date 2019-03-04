from matplotlib import pyplot as plt
import numpy as np
import pickle
import sys

try:
    pickle_file_location = sys.argv[1]
except:
    print('Usage: pythonw visualize.py path/to/pickle_file')

with open(pickle_file_location, 'rb') as pickle_file:
    data = pickle.load(pickle_file)

epochs = range(len(data))

steps_mean = [np.mean(data[episode]['steps']) for episode in data]
steps_std = [np.std(data[episode]['steps']) for episode in data]

scores_mean = [np.mean(data[episode]['scores']) for episode in data]
# scores_std = [np.std(data[episode]['scores']) for episode in data]

fig, ax1 = plt.subplots()
ax1.plot(epochs, steps_mean, c='r')
ax1.set_ylabel('mean steps', color='r')

ax2 = ax1.twinx()
ax2.plot(epochs, scores_mean, c='b')
ax2.set_ylabel('mean scores', color='b')

fig.tight_layout()
plt.savefig('figures/{}.pdf'.format(pickle_file_location.split('/')[-1].strip('.pickle')))
plt.show()
