import sys, os 
sys.path.append('/'.join(os.getcwd().split('/')[:4]))
from config.get import cfg
import pandas as pd
import numpy as np
from helper import check_and_create_dir
from data_processing.TokenStandardScaler import TokenStandardScaler
from sklearn.model_selection import train_test_split

def build_tensor(data):
    data = data.reset_index()
    cols = list(filter(lambda c: c not in ['cycle_id', 'token1','token2'], data.columns))
    N_TOKEN = 3 # cycle length
    K = len(cols)  # nb of features (eg. quote price & gasPrice)
    N = data.cycle_id.nunique() # number of cycles
    P = 600     # max time series length per cycle
    
    tensor = np.zeros((N, N_TOKEN,P, K))
    cycle_ids = np.zeros(N)
    i = 0
    def get_sorted_token_map(g):
        """Preserve token ordering (token at index j is the jth token in the cycle"""
        t = g[['token1','token2']].values
        u, ind = np.unique(t, return_index=True)
        u_sorted =  u[np.argsort(ind)]
        return dict(zip(u_sorted, range(len(u_sorted))))

    for cycle_id, group in iter(data.groupby('cycle_id')):
        token_map = get_sorted_token_map(group)
        cycle_ids[i] =cycle_id
        for _, g in iter(group.groupby(['token1','token2'])):
            a = g[cols].values 
            # zero padding
            padded = np.pad(a, [(0, P - len(a)),(0,0)])
            # assign and reshape into a matrix
            first_token = g.token1.iloc[0]
            token_ind = token_map[first_token]
            tensor[i,token_ind,:,:] = padded.reshape(1,P,K)
        i = i+1
    return cycle_ids[:-1],tensor[:-1]

def run(use_liquid = True , 
        nrows=10_000_000, 
        drop_columns=["time"],
        log_transformation=True,
        new_train_idx=True,
        skip_split=False,
        extra_dir = None,
        feature_name='ae',
        scaling = True,
        ):  
    # when files are loaded or store => add _liquid at the end of the name
    features_dir = 'liquid' if use_liquid else 'full'
    data_dir = cfg['files'][features_dir]
    ml_data_dir = cfg['directories'][features_dir]
    if extra_dir is not None:
        data_dir = data_dir[extra_dir]
        ml_data_dir = ml_data_dir[extra_dir]
    check_and_create_dir(ml_data_dir['ML_features'])

    cols = ["quotePrice","gasPrice"]
    print(f"loading data ...")
    if not skip_split:
        data = pd.read_csv(data_dir['preprocessed_data'],nrows=nrows)
        if drop_columns is not None:
            data = data.drop(columns=["time"])
        data = data.set_index(["cycle_id","token1","token2"])[cols]
        
        # train test split
        print("splitting")
        if new_train_idx:
            train_idx, test_idx = train_test_split(data.index.levels[0],train_size=0.8)
        else:
            idx_dir = cfg['files'][features_dir]
            train_idx, test_idx = np.load(idx_dir['train_ids']), np.load(idx_dir['test_ids'])
        X_train = data.loc[train_idx]
        X_test = data.loc[test_idx]

        X_train.reset_index().to_csv(data_dir['ae_train_features'])
        X_test.reset_index().to_csv(data_dir['ae_test_features'])  
        
        print(f"Shapes : X_train={X_train.shape}, X_test={X_test.shape}")
    else:
        # skip until end of spliting phase
        X_train = pd.read_csv(data_dir[f'{feature_name}_train_features']).drop(columns=['Unnamed: 0'])
        print(X_train.shape)
        X_test  = pd.read_csv(data_dir[f'{feature_name}_test_features']).drop(columns=['Unnamed: 0'])
    
    # log transformation
    if log_transformation:
        print(f"taking the log of all columns")
        X_train = np.log(X_train).dropna()   
        X_test = np.log(X_test).dropna()   
    # personal rescaling
    if scaling:
        scaler  = TokenStandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)
    print("building tensor")
    train_ids , train_tensor = build_tensor(X_train)
    test_ids  , test_tensor  = build_tensor(X_test)
    print(f"Shapes : train_tensor={train_tensor.shape}, test_tensor={test_tensor.shape}")

    print("Saving")
    np.save(data_dir[f'scaled_{feature_name}_train_features'] ,train_tensor)   
    np.save(data_dir[f'scaled_{feature_name}_test_features'] , test_tensor)
    np.save(data_dir['train_ids'] , train_ids)   
    np.save(data_dir['test_ids'] , test_ids)

if __name__ == "__main__":
    print("==== Run : build embedding features ====")
    run()
    print("==== Done ====")
