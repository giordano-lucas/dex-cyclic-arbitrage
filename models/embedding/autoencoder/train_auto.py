import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
plt.style.use('ggplot')
import sys 
import os
sys.path.append('/'.join(os.getcwd().split('/')[:4]))
from config.get import cfg
from tensorflow import keras
from tensorflow.keras import layers
import autoencoders

X_train = np.load(cfg['files']["liquid"]["scaled_ae_train_features"])
X_test  = np.load(cfg['files']["liquid"]["scaled_ae_test_features"])
print(f"shapes : X_train={X_train.shape},X_test={X_test.shape}")

model_name,autoencoder = autoencoders.CNN_fully_connected()
autoencoder.summary()
train_loss = []
test_loss = []

hist = autoencoder.fit(X_train, X_train,epochs=400,validation_data=(X_test, X_test),batch_size=16)

# save losses
train_loss += hist.history["loss"]
test_loss  += hist.history["val_loss"]


autoencoder.save(cfg["models"]["autoencoder"] + f"{model_name}")
np.save(file = cfg["models"]["autoencoder"]+ f"{model_name}_train_loss", arr = np.array(train_loss))
np.save(file = cfg["models"]["autoencoder"]+ f"{model_name}_test_loss", arr = np.array(test_loss))