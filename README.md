<img width="1665" alt="Screenshot 2021-12-29 at 09 52 48" src="https://user-images.githubusercontent.com/43466781/147819628-ed23274e-8d05-487d-b499-2b7a51e36eee.png">

Read our data story online [using the following link]() ! 

# Abstract

The paper Cyclic Arbitrage in Decentralized Exchange Markets[[1]](https://arxiv.org/abs/2105.02784) showed a large number of occurrences of cyclic arbitrages in decentralised exchanges compared to centralised ones. In their work, they mainly focus on analysing these cycles in terms of length, token distribution, daily patens and profitability.

However, the factors driving their appearance have not been studied yet.  To this end, we propose to extend the work of [[1]](https://arxiv.org/abs/2105.02784) on Uniswap data. Moreover, we also plan to study the predictive power of these factors in a binary classification setting. It will allow determining whether or not a cycle can actually be implemented and generate a positive return, which has an inherent market value.

# Goal

The goal of this project is to study exploited cyclic arbitrage in decentralised exchanges. We already have access to the [Cyclic transaction dataset](https://disco.ethz.ch/misc/uniswap/cycles_in_Uniswap.json) which contains cyclic arbitrages that were exploited. We intend to extract features out of events (trade rates, trade volumes, liquidity) preceding the arbitrages. 

These features could potentially be high dimensional (depending on the length of the time series) and we will need to use dimensionality reduction techinques to create an embedding to build a relevant set of features of our future machine learning models.

Then, we will cluster the arbitrages based on the computed features. Ideally, we would like to observe meaningful clusterings: profitable cycles get clustered together, cycles having similar duration (how long it is profitable) also end up in the same cluster, etc. Once meaningful clusters are obtained, it gets interesting to use the same features in a prediction model having profitability of the arbitrage as a target.

# Data Acquisition

In this study, we propose to use the following datasets 
1. [Cyclic transaction dataset](https://disco.ethz.ch/misc/uniswap/cycles_in_Uniswap.json) : dataset made available in [Cyclic Arbitrage in Decentralized Exchange Markets](https://arxiv.org/abs/2105.02784). It contains information about arbitrage cycles that were exploited on DEXes. 
2. [Uniswap rates preceeding cyclic transaction dataset](https://www.kaggle.com/ogst68/uniswap-rates-preceeding-cyclic-arbitrages-raw/download) : dataset gathered in this study. It contains  the rates and gas prices preceeding cycles (600 transaction for each token pair uniswap pool).

To obtain these datasets, please follow the instruction below: 
1. Run the script : `download_uniswap_cycles.sh`. It download the  `Cyclic transaction dataset`.
2. Download the `Uniswap rates preceeding cyclic transaction dataset` that was poseted on kaggle [here](https://www.kaggle.com/ogst68/uniswap-rates-preceeding-cyclic-arbitrages-raw/download) these data were previously fetched using the Bitquery platform (using the script `rates_from_Bitquery.py`). 

> **Note**: if you have access to the [IZAR EPFL cluster](https://www.epfl.ch/research/facilities/scitas/hardware/izar/), the simplest solution to get the datasets is to our `data` directory that was made available publicly under the following folder `scratch/izar/kapps/data/`.

## Data Processing
To process the downloaded data one need to follow these steps :
    1. run `filter_cycles.py` to filter cycles of  `cycles_in_Uniswap.json` on their lenghts. Give the desired lenght in argument (default is 6). The script produces a new dataset : `data/filtered_cycles.json`
    2. run `combine_raw_data.py` to combine the multiple fetched files available at `data/uniswap_raw_data/`. It produces a single .csv file (`data/uniswap_data.csv`)containing all rates and gas prices that were queried from bitQuery.   
    3. run `build_embedding_features.py` to create, scale and pad a train (`data/ML_features/raw_train_features.npy`) and test (`data/ML_features/raw_test_features.npy`) set out of `data/uniswap_data.csv`. These set a meant to be used to train an autoencoder for feature extraction.
    4. run `compute_embeddings.py` using the model we trained or yours (see models section) in argument to compute the embeddings of the previously generated train and test sets. The execution produces 2 new datasets : `data/ML_features/encoded_train_features.npy` and `data/ML_features/encoded_test_features.npy`
    5. run `build_prediction_data.py` to prepare the data needed for the prediction task. It produces train and test sets containing the profitability ofeach cycles as well as the tokens involved in the cycles.    

> **Note**: to simply the the data processing procedure for the reader, we created a shell script. In the base directory, run the following command:
```bash
bash scripts/data_processing.sh
```
# Methods

XXX

# Timeline and contributions :

## Week 1 : Data Acquisition & Setup (1)

| Task                                | Team member(s)                  | work hours  |
| :-----------------------------------|:--------------------------------| -----------:|
| Literature research                 | Lucas & Augustin                | 3h          |
| API choice and query design         | Lucas & Augustin                | 4h          |
| EPFL Cluster & environment setup    | Lucas                           | 2h          |
| Data fetching script test           | Augustin                        | 3h          |
| Data fetching validation            | Augustin                        | 2h          |
| Data fetching improvements          | Augustin                        | 2h          |


## Week 2 : Data preprocessing

| Task                                     | Team member(s)                  | work hours  |
| :----------------------------------------|:--------------------------------| -----------:|
| Data cleaning                            | Augustin                        | 5h          |
| Data exploration paper dataset           | Augustin                        | 2h          |
| Data exploration                         | Lucas                           | 3h          |
| Raw data => embedding format             | Lucas                           | 3h          |


## Week 3 : Embedding & Clustering 

| Task                                     | Team member(s)                  | work hours  |
| :----------------------------------------|:--------------------------------| -----------:|
| Autencoder keras basic code              | Lucas                           | 3h          |
| Comparision with PCA and debugging       | Lucas                           | 1h          |
| K-means                                  | Augustin                        | 2h          |

## Week 4 : Clustering analysis, Profitablity prediction & report writing

| Task                                    | Team member(s)                  | work hours  |
| :---------------------------------------|:--------------------------------| -----------:|
| Clustering analysis                     | Lucas                           | 4h          |
| Profitablity prediction setup           | Augustin                        | 2h          |
| Github pages setup                      | Lucas                           | 2h          |
| Data story (1)                          | Lucas                           | 5h          |
| Data story (2)                          | Augustin                        | 2h          |

### Week 5 : Improvements in data processing & report writing

| Task                                    | Team member(s)                  | work hours  |
| :---------------------------------------|:--------------------------------| -----------:|
| Token based scaling                     | Lucas & Augustin                |    5h        |
| Token one hot encoding                  | Lucas                           |    1h        |
| Token encoding in profitablity prediction   | Augustin                    |    1h        |
| Deep NN for   profitablity prediction   | Augustin                        |    1h        |
| Better data processing                  | Augustin                        |    2h        |
| Improved data exploration               | Lucas                           |    3h        |
| Better understanding of PCA output      | Augustin                        |    1h        |
| Autencoder testing                      | Augustin                        |    2h        |
| Data story (3)                          | Lucas                           |    1h        |        

### Week 6 : Hyperparameter opmisation, improvements & report writing

| Task                                    | Team member(s)                  | work hours  |
| :---------------------------------------|:--------------------------------| -----------:|
| Filter illiquid data                    | Lucas                           |    1h        |
| Research on attention learning          | Lucas                           |    2h        |
| Data processing simpler pipeline        | Augustin                        |    2h        |
| Autencoder improvement and debug        | Augustin                        |    3h        |
| Autencoder manual tests for several architectures        | Augustin       |    5h        |
| Talos setup                             | Lucas                           |    2h        |
| Hyperparameter opmisation               | Lucas & Augustin                |    4h        |
| Notebook comments and markdown          | Lucas & Augustin                |    3h        |
| Data story (4)                          | Lucas & Augustin                |    5h        |

## Total contribution:

| Team member                     | work hours   |
|:--------------------------------| ------------:|
| Lucas Giordano                  |    50h       |
| Augustin Kapps                  |    56h       |

# Notes to the reader
## Organisation of the repository

    .
    ├── data                                      # Data folder
    │ ├── uniswap_raw_data                        # data fetched from bitquery
    │ │  ├── uniswap_raw_data_0_1000.json.gz     # example of file
    │ │  ├── ...
    │ ├── additional_features_XXX.csv             #  data fetched from bitquery
    │ ├── cycles_in_Uniswap.json                  # dataset from the paper
    │ ├── filtered_cycles.json                    # only cycles of length 3 
    │ ├── uniswap_data.csv                        # csv version of the dataset fetched from biquery 
    │ ├── liquid_uniswap_data.csv                 # filter out illiquid token pair
    │ ├── additional_features_XXX.csv             # file used by the clustering and prediction task with extra features
    ├── data_acquisition                    # Scripts to fetch the datasets (from bitquery and from the paper)
    ├── data_exploration                    # Contains visualisations of the datasets
    ├── data_processing                     # All scripts to process the raw data into usables features for ML
    ├── models                              # all ML related tasks
    │ ├── clustering                        # files related to the clustering task
    │ ├── embedding                         # files related to the embedding task
    │ ├── prediction                        # files related to the profitablity prediction task
    ├── figures                             # Contains the ouput images and html used for the data story
    ├── requirements.txt                    # Dependencies
    └── README.md               
    
## How to run the code 

XXX

## Dependencies requirement

In the repository, we provide a `requirement.txt` file from which you can create a virutatal python environment.

> Note : to be able to import `talos` on the Scitas cluster, we need to update line 8 of `opt/venv-gcc/lib/python3.7/site-packages/kerasplotlib/traininglog.py` from `from keras.callbacks import Callback` to `from tensorflow.keras.callbacks import Callback`