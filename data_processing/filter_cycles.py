import json
import sys, os 
sys.path.append('/'.join(os.getcwd().split('/')[:4]))
from config.get import cfg
import pandas as pd
import numpy as np


def run(cycle_length = 6):
    
    raw_cycles = []
    for line in open(cfg['files']['all_cycles']):
        raw_cycles.append(line)
        
        
    f_out = open(cfg['files']['filtered_cycles'],"wt")
    cycle_id = 0
    for cycle in raw_cycles:
        cycle = json.loads(cycle)
        if len(cycle["path"])==cycle_length:
            cycle["cycle_id"] = cycle_id
            cycle_id+=1
            f_out.write(json.dumps(cycle)+"\n")
    f_out.close()
    print(cycle_id," cycles saved")    

if __name__ == "__main__":
    print("==== Run : filter cycles ====")
    run()
    print("==== Done ====")