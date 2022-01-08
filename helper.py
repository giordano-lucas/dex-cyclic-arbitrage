import os
import sys 
sys.path.append('/'.join(os.getcwd().split('/')[:4]))
from config.get import cfg

def check_and_create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
        
def load_cycles(mode="filtered"):
    if mode=="all":
        path = cfg['files']['all_cycles']
    else :
        path = cfg['files']['filtered_cycles'] 
    cycles = []
    for line in open(path):
        cycles.append(line)
    return cycles