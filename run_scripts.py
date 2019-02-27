import sys
import glob
import subprocess

try:
    script_folder = sys.argv[1]
except:
    print('Usage: python run_scripts.py path/to/scripts_folder')

for bash_file in glob.glob(script_folder + '*.sh'):
    subprocess.run(['sh', bash_file])
