import sys, os 
sys.path.append('/'.join(os.getcwd().split('/')[:4]))
from config.get import cfg
import pandas as pd
import numpy as np
from helper import check_and_create_dir
from sklearn.preprocessing import StandardScaler
from TokenStandardScaler import TokenStandardScaler
from sklearn.model_selection import train_test_split

<<<<<<< HEAD

def pad(X):
    def pad_index(index,P=600):
        padded_index = pd.MultiIndex.from_product([list(index.unique()),range(P)])
        as_frame = padded_index.to_frame()
        as_frame['cycle_id'], as_frame['token1'], as_frame['token2'] = zip(*as_frame[0])
        return as_frame.reset_index().drop(columns=["level_0","level_1",0]).set_index(["cycle_id","token1","token2",1]).index
    
    padded_index = pad_index(X.index,600)
    X[1] = X.reset_index().groupby(["cycle_id","token1","token2"]).cumcount().values
    return X.reset_index().set_index(["cycle_id","token1","token2",1]).reindex(padded_index,fill_value=0).dropna()


=======
>>>>>>> ae43ca54d6fa4c6ab045d83f1c4e1d4f9e2ecd99
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
        data = pd.read_csv(cfg['files'][features_dir]['liquid_preprocessed_data'],nrows=nrows)
    else :
        data = pd.read_csv(cfg['files'][features_dir]['preprocessed_data'],nrows=nrows)
        
    data = data.drop(columns=["time"]).set_index(["cycle_id","token1","token2"])[cols]
    

<<<<<<< HEAD
    print(f"taking the log of {cols}")
    data = np.log(data).dropna()


# def build_tensor(X_padded):
#     cycle_ids = []
#     tensor    = []
#     errors = 0
#     for cycle_id, group in iter(X_padded.groupby("cycle_id")): 
#             try :
#                 tensor.append(group[["quotePrice","gasPrice"]].values.reshape((3, 600, 2)))
#                 cycle_ids.append(cycle_id)
#             except:
#                 errors+=1
#     print(f"{errors} errors")   
#     return np.array(cycle_ids),np.array(tensor) 

def run():   
    data = pd.read_csv(cfg['files']['preprocessed_data'],nrows=10_000_000).drop(columns=["time"]) 
    data = data.set_index(["cycle_id","token1","token2"])

=======
#     print(f"taking the log of {cols}")
#     data = np.log(data).dropna()
    
>>>>>>> ae43ca54d6fa4c6ab045d83f1c4e1d4f9e2ecd99
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

<<<<<<< HEAD
    print("padding")
    train_padded = pad(tX_train)
    test_padded = pad(tX_test)
=======

    #print("padding")
    #train_padded = pad(tX_train)
    #test_padded = pad(tX_test)
    train_padded = tX_train
    test_padded  = tX_test
>>>>>>> ae43ca54d6fa4c6ab045d83f1c4e1d4f9e2ecd99
    print(f"Shapes : train_padded={train_padded.shape}, test_padded={test_padded.shape}")

    print("building tensor")
    train_ids , train_tensor = build_tensor(train_padded)
    test_ids  , test_tensor  = build_tensor(test_padded)
    print(f"Shapes : train_tensor={train_tensor.shape}, test_tensor={test_tensor.shape}")

    print("Saving")
    if use_liquid:
        np.save(cfg['files'][features_dir]['raw_test_features_liquid'] , test_tensor)
        np.save(cfg['files'][features_dir]['raw_train_features_liquid'] ,train_tensor)
        np.save(cfg['files'][features_dir]['test_ids_liquid'] , test_ids)
        np.save(cfg['files'][features_dir]['train_ids_liquid'] , train_ids)
    else : 
        np.save(cfg['files'][features_dir]['raw_test_features'] , test_tensor)
        np.save(cfg['files'][features_dir]['raw_train_features'] ,train_tensor)       
        np.save(cfg['files'][features_dir]['test_ids'] , test_ids)
        np.save(cfg['files'][features_dir]['train_ids'] , train_ids)

    
<<<<<<< HEAD

    np.save(cfg['files']['raw_test_features'] , test_tensor)
    np.save(cfg['files']['raw_train_features'] ,train_tensor)
    np.save(cfg['files']['test_ids'] , test_ids)
    np.save(cfg['files']['train_ids'] , train_ids)
    print("Done")
=======
>>>>>>> ae43ca54d6fa4c6ab045d83f1c4e1d4f9e2ecd99
if __name__ == "__main__":
    print("==== Run : build embedding features ====")
    check_and_create_dir(cfg['directories']['ML_features'])
    run()
    print("==== Done ====")