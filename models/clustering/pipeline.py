import sys 
sys.path.append("/scratch/izar/kapps/DEX-Cyclic-Arbitrage/")
from config.get import cfg
import tensorflow as tf
import numpy as np
from tensorflow import keras

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def extract_layers(main_model, starting_layer_ix, ending_layer_ix):
    """extract layers between starting_layer_ix and ending_layer_ix from a given model"""
     # create an empty model
    new_model = keras.Sequential()
    for ix in range(starting_layer_ix, ending_layer_ix + 1):
        curr_layer = main_model.get_layer(index=ix)
    # copy this layer over to the new model
        new_model.add(curr_layer)
    return new_model

def run(embedding_model_name,embedding_type="autoencoder"):
    print(f"evalutation {embedding_model_name} ({embedding_type})")
    
    # extract en encoder part of the autoencoder
    if embedding_type=="autoencoder":
        embedding_model_path = cfg["models"]["autoencoder"].format(embedding_model_name)
        autoencoder          = keras.models.load_model(embedding_model_path)
        encoder              = extract_layers(autoencoder,0,encoding_layer=5)
        
    if embedding_type=="PCA":
        embedding_model_path = cfg["models"]["PCA"].format(embedding_model_name)
        autoencoder          = None
        encoder              = None

    ##########################################################################
    print("==========================================")
    print("             loading data                 ")
    print("==========================================")
    train_raw = np.load(cfg["files"]["raw_train_features"])
    test_raw  = np.load(cfg["files"]["raw_test_features"])
    print("SHAPES:")
    print("     train raw features : ",train_raw.shape)
    print("     test raw features : ",test_raw.shape)
    ##########################################################################
    print("==========================================")
    print("                 encoding                 ")
    print("==========================================")
    train_encoded = encoder(train_raw).numpy()
    test_encoded = encoder(test_raw).numpy()

    n_train,_,d,_ = train_encoded.shape
    n_test,_,d,_ = test_encoded.shape

    train_encoded = train_encoded.reshape((n_train,d))
    test_encoded  = test_encoded.reshape((n_test,d))
    print("SHAPES:")
    print("     train encoded features : ",train_encoded.shape)
    print("     test encoded features : ",test_encoded.shape)

    ##########################################################################
    print("==========================================")
    print("                 scaling                  ")
    print("==========================================")

    scaler = StandardScaler()
    scaler.fit(train_encoded)
    X_train = scaler.transform(train_encoded)

    ##########################################################################
    print("==========================================")
    print("                 clustering               ")
    print("==========================================")

    silhouettes = []
    sse = []
    for k in range(2, k_max): # Try multiple k
        print(k,end="\r")
        # Cluster the data and assign the labels
        kmeans =  KMeans(n_clusters=k, random_state=42)
        labels =  kmeans.fit_predict(X_train)
        # Get the Silhouette score
        score = silhouette_score(X_train, labels)
        silhouettes.append({"k": k, "score": score})

        sse.append({"k": k, "sse": kmeans.inertia_})

    # Convert to dataframes
    silhouettes = pd.DataFrame(silhouettes)
    sse = pd.DataFrame(sse)

    # merge scores
    scores = sse.merge(silhouettes,on='k')

    scores.to_csv(f"{embedding_model_name}_scores.csv")

run("autoencoder_model_simple",embedding_type="autoencoder")