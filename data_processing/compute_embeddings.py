
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
    features_dir = 'liquid' if use_liquid else 'full'
    autoencoder = keras.models.load_model(cfg["models"]["autoencoder"]+model_name)
    encoding_layer = int(models_cfg["encoding_layer"][model_name])
    # extract en encoder part of the autoencoder
    encoder = extract_layers(autoencoder,0,encoding_layer)
    encoder.summary()
    
    print("loading data")
    train_ae = np.load(cfg["files"][features_dir]["scaled_ae_train_features"])
    test_ae  = np.load(cfg["files"][features_dir]["scaled_ae_test_features"])

    #print(encoder(train_ae[:2]))
    print("SHAPES:")
    print("     train ae features : ",train_ae.shape)
    print("     test ae features : ",test_ae.shape)
    print("==========================================")
    print("                 encoding                 ")
    print("==========================================")
    train_encoded = encoder(train_ae).numpy()
    test_encoded = encoder(test_ae).numpy()


    print("SHAPES:")
    print("     train encoded features : ",train_encoded.shape)
    print("     test encoded features : ",test_encoded.shape)
    print("==========================================")
    print("                 saving                 ")
    print("==========================================")
    np.save(cfg['files'][features_dir]['encoded_train_features'] , train_encoded)
    np.save(cfg['files'][features_dir]['encoded_test_features'] ,test_encoded)
    print("Done!")

if __name__ == "__main__":
    print("==== Run : compute embedding ====")
    compute()
    print("==== Done ====")
