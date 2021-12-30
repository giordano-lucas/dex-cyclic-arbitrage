import sys 
sys.path.append("/scratch/izar/kapps/DEX-Cyclic-Arbitrage/")
from config.get import cfg
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
print("===================================")     
data = pd.read_csv(cfg['files']['preprocessed_data'],nrows=25_000_000)

# from data exploration log was found to be a good normaliser
def log_scaling(data):
    data['baseAmount']  = np.log(data.baseAmount)
    data['quoteAmount'] = np.log(data.quoteAmount)
    data['quotePrice']  = np.log(data.quotePrice)
    return data;

log_data = log_scaling(data).dropna()
print(len(log_data)," rows loaded")
N_TOKEN = 3 # cycle length
K = 2       # quote price & gasPrice
N =data.cycle_id.nunique() # number of cycles
P = 600     # max time series length per cycle

def build_tensor(data):
    tensor = np.zeros((N, N_TOKEN,P, K))
    cycle_ids = np.zeros(N)
    i = 0
    def get_sorted_token_map(g):
        t = g[['token1','token2']].values
        u, ind = np.unique(t, return_index=True)
        u_sorted =  u[np.argsort(ind)]
        return dict(zip(u_sorted, range(len(u_sorted))))

    for cycle_id, group in iter(data.groupby('cycle_id')):
        token_map = get_sorted_token_map(group)
        cycle_ids[i] =cycle_id
        for _, g in iter(group.groupby(['token1','token2'])):
            a = g[['quotePrice','gasPrice']].values 
            # zero padding
            padded = np.pad(a, [(0, P - len(a)),(0,0)])
            # assign and reshape into a matrix
            first_token = g.token1.iloc[0]
            token_ind = token_map[first_token]
            tensor[i,token_ind,:,:] = padded.reshape(1,P,K)
        i = i+1
    return cycle_ids[:-1],tensor[:-1]
   
print("Processing")
cycle_ids,X = build_tensor(log_data)
print("data shape : ",X.shape)
print("ids shape : ",cycle_ids.shape)
print("Saving")
np.save(cfg['files']['raw_features'],X)
np.save(cfg['files']['cycle_ids'],cycle_ids)
print("Done")