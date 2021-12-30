import sys 
sys.path.append("/scratch/izar/kapps/DEX-Cyclic-Arbitrage/")
from config.get import cfg
import tensorflow as tf
import numpy as np
from tensorflow import keras

def mean_squared_error(x,y):
    return np.sum((x-y)**2)/x.size

# load model
model_name  = "autoencoder_model_simple" #autoencoder_model_simple
autoencoder = keras.models.load_model(model_name)

# load data

tX_train = np.load(cfg['files']['raw_train_features'])
tX_test  = np.load(cfg['files']['raw_test_features'])
N_train, N_test = tX_train.shape[0], tX_test.shape[0]
ttX_train = tX_train.reshape(N_train,-1)
ttX_test  = tX_test.reshape(N_test,-1)

ae_train = mean_squared_error(ttX_train,autoencoder.predict(tX_train).reshape(N_train,-1))
ae_test  = mean_squared_error(ttX_test , autoencoder.predict(tX_test).reshape(N_test,-1))
print(f"MSE train : {ae_train} \nMSE test  : {ae_test}")

# PCA 

from sklearn.decomposition import PCA
pca = PCA(n_components=100)
pca.fit(ttX_train)
pca_train = mean_squared_error(ttX_train, pca.inverse_transform(pca.transform(ttX_train)))
pca_test =  mean_squared_error(ttX_test, pca.inverse_transform(pca.transform(ttX_test)))
print(f"MSE train : {pca_train} \nMSE test  : {pca_test}")

# store results in dataframe

perf = pd.DataFrame(
    { 
        'MSE': [ae_train, ae_test, pca_train, pca_test],
        'model': ['Autoencoder','Autoencoder', 'PCA','PCA'],
        'set': ['train set','test set','train set','test set']
    })
print(perf)
# save figue
import plotly.express as px
fig = px.bar(perf, x="set", y="MSE",
             color='model', barmode='group',
             height=400, range_y=(2500,3000))
fig.show()