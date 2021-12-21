import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
%load_ext autoreload
%autoreload 2
plt.style.use('ggplot')
from tensorflow import keras
from tensorflow.keras import layers
import sys 
sys.path.append("/scratch/izar/kapps/DEX-Cyclic-Arbitrage/")
from config.get import cfg

X = np.load(cfg['files']["raw_features"])

# train/test split and standard scaling 
test_size = 0.3
N_train = int(N * (1 - test_size))
X_train, X_test, train_id, test_id = train_test_split(X, np.arange(X.shape[0]),test_size=test_size, random_state=123)


#scaler = StandardScaler()
#scaler.fit(X_train)

#X_train_scaled = scaler.transform(X_train)
#X_test_scaled  = scaler.transform(X_test)

X_train_scaled = X_train
X_test_scaled = X_test
print("=== SHAPES ====")
print(f"X_train_scaled:{X_train_scaled.shape},X_test_scaled:{X_test_scaled.shape}")

np.save(cfg['files']['raw_test_features'] , X_test)
np.save(cfg['files']['raw_train_features'] ,X_train)
np.save(cfg['files']['raw_test_ids'] , test_id)
np.save(cfg['files']['raw_train_ids'] , train_id)


def build_model():
    input_img = keras.Input(shape=(3,600, 2))
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((3, 3), padding='same')(x)
    x = layers.Conv2D(4, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    encoded = layers.Conv2D(1, (2, 2), activation='relu', padding='same')(x)
    # at this point the representation is 100-dimensional
    x = layers.Conv2D(4, (2, 2), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((1, 2))(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((3, 3))(x)
    decoded = layers.Conv2D(2, (3, 3), activation='relu', padding='same')(x)

    autoencoder = keras.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', metrics=['accuracy'],loss='mean_squared_error',)
    
    return autoencoder


autoencoder = build_model()
autoencoder.summary()



hist = autoencoder.fit(X_train_scaled, X_train_scaled,
                epochs=100,
                batch_size=32,
                shuffle=True,
                verbose=1)

autoencoder.save('autoencoder_model')


plt.plot(hist.history["loss"])
plt.xlabel("epoch")
plt.ylable("loss")
plt.savefig("loss.png")




