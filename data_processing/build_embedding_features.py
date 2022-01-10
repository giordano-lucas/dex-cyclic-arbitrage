import sys, os 
sys.path.append('/'.join(os.getcwd().split('/')[:4]))
from config.get import cfg
import pandas as pd
import numpy as np
from helper import check_and_create_dir
from sklearn.preprocessing import StandardScaler
from TokenStandardScaler import TokenStandardScaler
from sklearn.model_selection import train_test_split

def build_tensor(data):
    data = data.reset_index()
    
    N_TOKEN = 3 # cycle length
    K = 2       # quote price & gasPrice
    N =data.cycle_id.nunique() # number of cycles
    P = 600     # max time series length per cycle
    
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
            if (len(a) < 300):
                print("eeeeerrrrrrroorror")
            padded = np.pad(a, [(0, P - len(a)),(0,0)])
            # assign and reshape into a matrix
            first_token = g.token1.iloc[0]
            token_ind = token_map[first_token]
            tensor[i,token_ind,:,:] = padded.reshape(1,P,K)
        i = i+1
    return cycle_ids[:-1],tensor[:-1]


def run(use_liquid = True ,nrows=10_000_000):  
    # when files are loaded or store => add _liquid at the end of the name
    features_dir = 'liquid' if use_liquid else 'full'
    cols = ["quotePrice","gasPrice"]
    print("loading data")
    if use_liquid:
        data = pd.read_csv(cfg['files'][features_dir]['preprocessed_data'],nrows=nrows)
    else :
        data = pd.read_csv(cfg['files'][features_dir]['preprocessed_data'],nrows=nrows)
        
    data = data.drop(columns=["time"]).set_index(["cycle_id","token1","token2"])[cols]
    

    print(f"taking the log of {cols}")
    data = np.log(data).dropna()
    
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


    #print("padding")
    #train_padded = pad(tX_train)
    #test_padded = pad(tX_test)
    train_padded = tX_train
    test_padded  = tX_test
    print(f"Shapes : train_padded={train_padded.shape}, test_padded={test_padded.shape}")

    print("building tensor")
    train_ids , train_tensor = build_tensor(train_padded)
    test_ids  , test_tensor  = build_tensor(test_padded)
    print(f"Shapes : train_tensor={train_tensor.shape}, test_tensor={test_tensor.shape}")

    print("Saving")
    if use_liquid:
        np.save(cfg['files'][features_dir]['raw_test_features'] , test_tensor)
        np.save(cfg['files'][features_dir]['raw_train_features'] ,train_tensor)
        np.save(cfg['files'][features_dir]['test_ids'] , test_ids)
        np.save(cfg['files'][features_dir]['train_ids'] , train_ids)
    else : 
        np.save(cfg['files'][features_dir]['raw_test_features'] , test_tensor)
        np.save(cfg['files'][features_dir]['raw_train_features'] ,train_tensor)       
        np.save(cfg['files'][features_dir]['test_ids'] , test_ids)
        np.save(cfg['files'][features_dir]['train_ids'] , train_ids)

    
if __name__ == "__main__":
    print("==== Run : build embedding features ====")
    check_and_create_dir(cfg['directories']['ML_features'])
    run()
    print("==== Done ====")