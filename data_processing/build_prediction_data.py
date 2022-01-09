import json
import requests
import time 
import gzip
import sys,os 
sys.path.append('/'.join(os.getcwd().split('/')[:4]))
from helper import *
from config.get import cfg
import pandas as pd
import numpy as np


def run():
    print("Loading filtered cycles...")
    # load filtered cycles
    filtered_cycles = load_cycles("filtered")
    
    # extract relevant data
    cycle_ids = []
    revenues = []
    costs = []
    token1 = []
    token2 = []
    token3 = []
    for line in filtered_cycles:
        line = json.loads(line)
        cycle_ids.append(line["cycle_id"])
        revenues.append(line["revenue"])
        costs.append(line["cost"])
        token1.append(line["path"][0])
        token2.append(line["path"][2])
        token3.append(line["path"][4])

    # reformat
    features = pd.DataFrame({
        "cycle_id":cycle_ids,
        "revenues":revenues,
        "costs":costs,
        "token1":token1,
        "token2":token2,
        "token3":token3
        })
    features = features.astype({'cycle_id': 'int32',"revenues":np.float64,"costs":np.float64})

    features["profits"] = features["revenues"]-features["costs"] 
    features["profitability"] = features["profits"]>0

    #features.to_csv(cfg["files"]["features"])
    
    # load the train and test ids
    train_ids = np.load(cfg['files']['train_ids']).astype(int)
    test_ids  = np.load(cfg['files']['test_ids']).astype(int)
    train_ids = pd.DataFrame({"cycle_id":train_ids})
    test_ids  = pd.DataFrame({"cycle_id":test_ids})

    features_i = features.set_index('cycle_id')
    
    # split the extracted data
    f_train = train_ids.join(features_i,on="cycle_id",lsuffix="_")
    f_test  = test_ids.join(features_i,on="cycle_id",lsuffix="_")
    
    # save the extracted data as a train and test sets
    f_train.to_csv(cfg["files"]["additional_features_train"])
    f_test.to_csv(cfg["files"]["additional_features_test"])

if __name__ == "__main__":
    print("==== Run : build prediction data ====")
    run()
    print("==== Done ====")