import sys
import os
import subprocess

try:
    script_folder = sys.argv[1]
except:
    print('Usage: python run_scripts.py path/to/scripts_folder')

for bash_file in os.listdir(script_folder):
    subprocess.run(['sbatch', script_folder + '/'  + bash_file])
