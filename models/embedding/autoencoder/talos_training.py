import numpy as np
import pandas as pd
import sys, os 
sys.path.append('/'.join(os.getcwd().split('/')[:4]))
from config.get import cfg
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.activations import relu, elu
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
import helper
import autoencoders
import talos
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# fit function for talos
def fit_model(x_train, y_train, x_val, y_val, params):
    # build model
    model_name,autoencoder = autoencoders.talos_architecture(params)    
    # fit model
    out = autoencoder.fit(
        x_train, y_train,
        shuffle=True,
        verbose=1,
        batch_size=params['batch_size'],
        epochs=params['epochs'],
        validation_data=(x_val, y_val),
        callbacks=[talos.utils.early_stopper(params['epochs'])],
    )
    fig = px.line(x=range(params['epochs']), y=out.history["loss"], title=f'MSE Loss for model : {model_name}')
    fig.show()
    return out, autoencoder

def train():
    X_train = np.load(cfg['files']["liquid"]["raw_train_features"])
    X_test  = np.load(cfg['files']["liquid"]["raw_test_features"])
    
    print(f"Prencentage of padded zero in the training set : {100* np.mean(np.isclose(X_train, 0.0)): 2.2f} %")
    print(f"Prencentage of padded zero in the test set     : {100* np.mean(np.isclose(X_test, 0.0)): 2.2f} %")



    ## the parameter that you want to be optimized are defined in this dictionnary
    p = {
        'activation':['selu', 'elu'],
        'dense_layers' : [1,3,5],
        'first_neuron': [200,300,500],
        'dropout': [0, .25, .5],
        'batch_size': [16,32],
        'optimizer': ['adam', 'nadam'],
        'epochs': [70,120]  
    }

    scan_object = talos.Scan(
        x=X_train, y=X_train, 
        params=p, 
        model=fit_model, 
        experiment_name='talos-autoencoder', 
        val_split=0.2,
        minimize_loss=True,
    ) 
    
    
if __name__ == "__main__":
    print("==== Run : build embedding features ====")
    train()
    print("==== Done ====")