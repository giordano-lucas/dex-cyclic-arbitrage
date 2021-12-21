import sys 
sys.path.append("/scratch/izar/kapps/DEX-Cyclic-Arbitrage/")
from config.get import cfg
import tensorflow as tf
import numpy as np
from tensorflow import keras

autoencoder = keras.models.load_model(cfg["models"]["autoencoder"])

encoding_layer = 5

def extract_layers(main_model, starting_layer_ix, ending_layer_ix):
    """extract layers between starting_layer_ix and ending_layer_ix from a given model"""
     # create an empty model
    new_model = keras.Sequential()
    for ix in range(starting_layer_ix, ending_layer_ix + 1):
        curr_layer = main_model.get_layer(index=ix)
    # copy this layer over to the new model
        new_model.add(curr_layer)
    return new_model
# extract en encoder part of the autoencoder
encoder = extract_layers(autoencoder,0,encoding_layer)

encoder.summary()

train_raw = np.load(cfg["files"]["raw_train_features"])
test_raw  = np.load(cfg["files"]["raw_test_features"])
print(encoder(train_raw[:2]))
# print("SHAPES:")
# print("     train raw features : ",train_raw.shape)
# print("     test raw features : ",test_raw.shape)
# print("==========================================")
# print("                 encoding                 ")
# print("==========================================")
# train_encoded = encoder(train_raw).numpy()
# test_encoded = encoder(test_raw).numpy()

# n_train,_,d,_ = train_encoded.shape
# n_test,_,d,_ = test_encoded.shape

# train_encoded = train_encoded.reshape((n_train,d))
# test_encoded  = test_encoded.reshape((n_test,d))
# print("SHAPES:")
# print("     train encoded features : ",train_encoded.shape)
# print("     test encoded features : ",test_encoded.shape)
# print("==========================================")
# print("                 saving                 ")
# print("==========================================")
# np.save(cfg['files']['encoded_train_features'] , train_encoded)
# np.save(cfg['files']['encoded_test_features'] ,test_encoded)
# print("Done!")