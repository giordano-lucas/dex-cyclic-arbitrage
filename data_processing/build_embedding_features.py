import sys 
sys.path.append("/scratch/izar/kapps/DEX-Cyclic-Arbitrage/")
from config.get import cfg
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from TokenStandardScaler import TokenStandardScaler
from sklearn.model_selection import train_test_split


def pad(X):
    def pad_index(index,P=600):
        padded_index = pd.MultiIndex.from_product([list(index.unique()),range(P)])
        as_frame = padded_index.to_frame()
        as_frame['cycle_id'], as_frame['token1'], as_frame['token2'] = zip(*as_frame[0])
        return as_frame.reset_index().drop(columns=["level_0","level_1",0]).set_index(["cycle_id","token1","token2",1]).index
    
    padded_index = pad_index(X.index,600)
    X[1] = X.reset_index().groupby(["cycle_id","token1","token2"]).cumcount().values
    return X.reset_index().set_index(["cycle_id","token1","token2",1]).reindex(padded_index,fill_value=0)


def build_tensor(X_padded):
    cycle_ids = []
    tensor    = []
    i = 0
    for cycle_id, group in iter(X_padded.groupby("cycle_id")): 
            try :
                tensor.append(group[["quotePrice","gasPrice"]].values.reshape((3, 600, 2)))
                cycle_ids.append(cycle_id)
            except:
                print(i)
            i+=1
            
    return np.array(cycle_ids),np.array(tensor) 

def run():
    print("=========================")     
    data = pd.read_csv(cfg['files']['preprocessed_data'],nrows=10_000_000).drop(columns=["time"])
    data = data.set_index(["cycle_id","token1","token2"])


    # train test split
    print("splitting")
    train_ix, test_ix = train_test_split(data.index.levels[0],train_size=0.8)
    X_train = data.loc[train_ix]
    X_test = data.loc[test_ix]
    print(f"Shapes : X_train={X_train.shape}, X_test={X_test.shape}")
    # personal rescaling
    scaler   = TokenStandardScaler()
    tX_train = scaler.fit_transform(X_train)
    tX_test  = scaler.transform(X_test)



    print("padding")
    train_padded = pad(tX_train)
    test_padded = pad(tX_test)
    print(f"Shapes : train_padded={train_padded.shape}, test_padded={test_padded.shape}")

    print("building tensor")
    train_ids , train_tensor = build_tensor(train_padded)
    test_ids  , test_tensor  = build_tensor(test_padded)
    print(f"Shapes : train_tensor={train_tensor.shape}, test_tensor={test_tensor.shape}")



    print("Saving")
    np.save(cfg['files']['raw_test_features'] , test_tensor)
    np.save(cfg['files']['raw_train_features'] ,train_tensor)
    np.save(cfg['files']['test_ids'] , test_ids)
    np.save(cfg['files']['train_ids'] , train_ids)
    print("Done")