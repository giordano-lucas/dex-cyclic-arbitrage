import pandas as pd
import numpy as np
import sys, os
sys.path.append('/'.join(os.getcwd().split('/')[:4]))
from config.get import cfg
import gc
# read data generated by data_processing/build_embedding_features.py
X_train = pd.read_csv(cfg['files']["liquid"]['ae_train_features']).drop(columns=['Unnamed: 0'])
X_test  = pd.read_csv(cfg['files']["liquid"]['ae_test_features']).drop(columns=['Unnamed: 0'])

def logret(data, cols = ['quotePrice', 'gasPrice']):
    """Constructs log return series for quotePrice and gasPrice"""
    fcols = []
    for col in cols:
        ncol = f'logret_{col}'
        data[ncol] = np.log(data[col])
        fcols.append(ncol)
    
    grouped = data.groupby(['cycle_id', 'token1','token2'])[fcols]
    log_ret = grouped.diff()
    data.drop(columns=fcols,inplace=True)
    return log_ret

def get_ta_data(data, window):
    """Compute technical analysis indicators (mean,std) for the given window"""
    g = data.groupby(['cycle_id', 'token1','token2']).rolling(window).agg({
                     'quotePrice'        : ['mean','std'],
                     'gasPrice'          : ['mean','std'],
                     'logret_quotePrice' : ['mean','std'],
                     'logret_gasPrice'   : ['mean','std'], 
            })
    new_cols = [f'quotePrice_{window}'       ,f'gasPrice_{window}',
                f'logret_quotePrice_{window}',f'logret_gasPrice_{window}' ]
    g.columns.set_levels(new_cols,level=0,inplace=True)
    return g.fillna(0.0)

def pipeline(data):
    """Compute logreturns and other TA indicators"""
    # compute log return times series
    logret_data = logret(data)
    # add these series into the original dataframe
    for c in logret_data.columns:
        data[c] = logret_data[c]
    del logret_data
    # compute the indicators for 5 and 20 window
    print("Compute for window 1 / 2")
    ta_data_20 = get_ta_data(data, 20)
    print("Compute for window 2 / 2")
    ta_data_5 = get_ta_data(data , 5)
    # concatenate ta columns
    ruled_based_embedding = pd.concat((ta_data_5, ta_data_20),axis=1)
    del ta_data_20
    del ta_data_5
    gc.collect()
    # reshape multiIndex columns => simple index 
    ruled_based_embedding.columns = ["_".join(a) for a in ruled_based_embedding.columns.to_flat_index()]
    return ruled_based_embedding.reset_index()

def run():
    print("=== Build encoded ruled based data for train set ===")
    ta_train = pipeline(X_train)
    ta_train.to_csv(cfg['files']['liquid']['ruled_based']['encoded_train_features'])
    del ta_train
    gc.collect()
    print("=== Build encoded ruled based data for test set ===")
    ta_test = pipeline(X_test)
    ta_test.to_csv(cfg['files']['liquid']['ruled_based']['encoded_test_features'])


