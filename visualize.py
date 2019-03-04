import pickle
import sys


try:
    pickle_file_location = sys.argv[1]
except:
    print('Usage: pythonw visualize.py path/to/pickle_file')

with open(pickle_file_location, 'rb') as pickle_file:
    data = pickle.load(pickle_file)
    print(data.keys())
