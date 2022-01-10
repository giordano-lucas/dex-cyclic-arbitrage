
import sys 
import os 
import sys, os 
sys.path.append('/'.join(os.getcwd().split('/')[:4]))
from config.get import cfg,models_cfg
import tensorflow as tf
import numpy as np
from tensorflow import keras


def extract_layers(main_model, starting_layer_ix, ending_layer_ix):
    """extract layers between starting_layer_ix and ending_layer_ix from a given model"""
     # create an empty model
    new_model = keras.Sequential()
    for ix in range(starting_layer_ix, ending_layer_ix + 1):
        curr_layer = main_model.get_layer(index=ix)
    # copy this layer over to the new model
        new_model.add(curr_layer)
    return new_model


def compute(model_name = "fully_connected_3L", use_liquid=True):
    autoencoder = keras.models.load_model(cfg["models"]["autoencoder"]+model_name)
    encoding_layer = int(models_cfg["encoding_layer"][model_name])
    # extract en encoder part of the autoencoder
    encoder = extract_layers(autoencoder,0,encoding_layer)
    encoder.summary()
    
    if use_liquid:
        print("loading liquid data")
        train_raw = np.load(cfg["files"]["liquid"]["raw_train_features_liquid"])
        test_raw  = np.load(cfg["files"]["liquid"]["raw_test_features_liquid"])
    else : 
        print("loading all data (liquid+illiquid)")
        train_raw = np.load(cfg["files"]["full"]["raw_train_features"])
        test_raw  = np.load(cfg["files"]["full"]["raw_test_features"])
    #print(encoder(train_raw[:2]))
    print("SHAPES:")
    print("     train raw features : ",train_raw.shape)
    print("     test raw features : ",test_raw.shape)
    print("==========================================")
    print("                 encoding                 ")
    print("==========================================")
    train_encoded = encoder(train_raw).numpy()
    test_encoded = encoder(test_raw).numpy()


    print("SHAPES:")
    print("     train encoded features : ",train_encoded.shape)
    print("     test encoded features : ",test_encoded.shape)
    print("==========================================")
    print("                 saving                 ")
    print("==========================================")
    if use_liquid:
        np.save(cfg['files']["liquid"]['encoded_train_features'] , train_encoded)
        np.save(cfg['files']["liquid"]['encoded_test_features'] ,test_encoded)
    else : 
        np.save(cfg['files']["full"]['encoded_train_features'] , train_encoded)
        np.save(cfg['files']["full"]['encoded_test_features'] ,test_encoded)
    print("Done!")

if __name__ == "__main__":
    print("==== Run : compute embedding ====")
    compute()
    print("==== Done ====")
