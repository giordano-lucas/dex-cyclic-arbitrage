# Autoencoder
---

# Scripts and notebooks : 
This folder is dedicated to the study of autoencoder models for embedding computation. We focus here on obtaining a dimension reduction method having the smallest reconstruction error (MSE) focusing on autoencoders.	The study is divided into following files :

* `autoencoders.py`:Python file containing the architecture of the autoencoders to train

* `models training.ipynb`:notebook where the complex models (CNN and fully connected) are trained 

* `linear_vs_PCA.ipynb` : Note where the linear model is trained and compared to PCA. 

* `losses comparison.ipynb`: Each of the previously described notebook save the models and their losses into the `results` folder. This notebook is meant to visualize the produced losses.

* `talos_training.py`: Python script that runs `Talos`on a given set of parameters to tune. `Talos` is a library for architecture testing. It allows us to train and test multiple architectures of neural networks.It saves the results (architecture: test loss) into the Talos-autoencoder folder 

* `compute_embeddings.py` python script that uses one of the trained models to compute embeddings of given datasets.
The results are saved in the following folders. The script takes the name of the autoencoder to use as input, the default value is: `fully_connected_3L`. 

# Result folders

* `results`: Resulting models trained by `models training.ipynb` and `linear_vs_PCA.ipynb` . It contains the trained models and their losses.

* `talos-autoencoder`: Results of architecture trained  and tested by `Talos`
