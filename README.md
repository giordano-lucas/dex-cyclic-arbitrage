<img width="1665" alt="Screenshot 2021-12-29 at 09 52 48" src="https://user-images.githubusercontent.com/43466781/147819628-ed23274e-8d05-487d-b499-2b7a51e36eee.png">

:point_right: Read our **data story** online [using the following link](https://giordano-lucas.github.io/dex-cyclic-arbitrage/) :rocket: 

# Abstract

The paper Cyclic Arbitrage in Decentralized Exchange Markets[[1]](https://arxiv.org/abs/2105.02784) showed a large number of occurrences of cyclic arbitrages in decentralised exchanges compared to centralised ones. In their work, they mainly focus on analysing these cycles in terms of length, token distribution, daily patens and profitability.

However, the factors driving their appearance have not been studied yet.  To this end, we propose to extend the work of [[1]](https://arxiv.org/abs/2105.02784) on Uniswap data. Moreover, we also plan to study the predictive power of these factors in a binary classification setting. It will allow determining whether or not a cycle can actually be implemented and generate a positive return, which has an inherent market value.

# Goal

The goal of this project is to study exploited cyclic arbitrage in decentralised exchanges. We already have access to the [Cyclic transaction dataset](https://disco.ethz.ch/misc/uniswap/cycles_in_Uniswap.json) which contains cyclic arbitrages that were exploited. We intend to extract features out of events (trade rates, trade volumes, liquidity) preceding the arbitrages. 

These features could potentially be high dimensional (depending on the length of the time series) and we will need to use dimensionality reduction techinques to create an embedding to build a relevant set of features of our future machine learning models.

Then, we will cluster the arbitrages based on the computed features. Ideally, we would like to observe meaningful clusterings: profitable cycles get clustered together, cycles having similar duration (how long it is profitable) also end up in the same cluster, etc. Once meaningful clusters are obtained, it gets interesting to use the same features in a prediction model having profitability of the arbitrage as a target.

# Methods

1. **Data preprocessing**: 

    1. Keep only cycles of length 3.
    2. Filter out illiquid tokens.
    3. Log-transformation for heavy-tailed features
    4. Token-based standard scaling
    5. Zero padding for length standardisation.

2. **Cycles embedding**:

    1. After preprocessing an autencoder is built.
    2. Multiple architectures are tested (linear, multilayer densly connected, convolutional layer). 
    3. Their performance is compared to a classical PCA approach. 
    4. In part 4. Profitablity prediction, the performance of the different embeddings techniques is evaluated on the accuracy of the task.

3. **Cycles clustering**:

    1. Use the embedding, a KMeans clustering is constructed. 
    2. Clusters in the training set are analysed 
    3. Based on the test set results, we can understand whether or not there is predictibility in the results obtained in point 2.

4. **Cycle profitablity prediction**:

    1. Study profitability prediction for arbitrage cycles.
    2. Multiples models are tested (logistic regression, SVM).
    3. The impact of adding token encoding to the models is tested.
    4. The performance of the different embeddings is evaluated.

# Notes to the reader

Each folder contains a decidaced `README` where extra instruction and details are given.

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

1. Follow the steps in [Data Acquisition](data_acquisition/README.md) to download the raw datasets 2. Follow the steps in [Data Processing](data_processing/README.md) to generate the preprocessed data
2. **Data exploration**: run the `data_exploration/data_exploration.ipynb` notebook to see the data exploration steps taken.
3. **Embeddings**: open the `models/embedding` folder:
    1. Autoencoder: follow the steps in [Build Rule-based features](models/embedding/autoencoder/README.md) to understand how to train the autoencoders
    2. PCA: run the `pca_embedding.ipynb` notebook to create the `PCA` embedding.
    3. Rule-based: follow the steps in [Build Rule-based features](models/embedding/rule_based/README.md) to generate preprocessed data usefull for performance comparision.
4. **Clustering**: run the `models/clustering/Kmeans.ipynb` notebook to see the code related to the clustering.
5. **Profitablity prediction**: run the `models/prediction/prediction.ipynb` notebook for the profitablity prediction task.

## Dependencies requirement

In the repository, we provide a `requirement.txt` file from which you can create a virtual python environment.

## Side note on the Scitas Cluster

If you want to run our code in the scitas cluster, you will need several additional steps for the  set-up:

1. Create a compatible Jupyter/Tensorflow environment using the [following official tutotrial](https://scitas-data.epfl.ch/confluence/display/DOC/How+to+use+Jupyter+and+Tensorflow+on+Izar)
2. To be able to import `talos` on the Scitas cluster, we need to update line 8 of `opt/venv-gcc/lib/python3.7/site-packages/kerasplotlib/traininglog.py` from `from keras.callbacks import Callback` to `from tensorflow.keras.callbacks import Callback`

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
| Add ruled based indicators for autoencoder performance comparision | Lucas |    2h       |        

### Week 6 : Hyperparameter opmisation, improvements & report writing

| Task                                    | Team member(s)                  | work hours   |
| :---------------------------------------|:--------------------------------| ------------:|
| Filter illiquid data & debug            | Lucas                           |    3h        |
| Update architecture for liquid data     | Augustin                        |    3h        |
| Research on attention learning          | Lucas                           |    2h        |
| Data processing simpler pipeline        | Augustin                        |    2h        |
| Autencoder improvement and debug        | Augustin                        |    3h        |
| Autencoder manual tests for several architectures        | Augustin       |    8h        |
| Talos setup                             | Lucas                           |    2h        |
| Hyperparameter opmisation               | Lucas & Augustin                |    4h        |
| Kmeans : better silouhette analysis     | Lucas                           |    1h        |
| Kmeans : update results for liquid data | Lucas                           |    3h        |
| Ruled based data : pandas-ta implementation    | Lucas                    |    1h        |
| Ruled based data : pandas implementation       | Lucas                    |    3h        |
| Ruled based data : code optimisation           | Lucas                    |    3h        |
| Ruled based data : performance comparision with AE  | Lucas               |    1h        |
| Notebook comments and markdown          | Lucas & Augustin                |    4h        |
| Data story (4)                          | Lucas & Augustin                |    6h        |

## Total contribution:

| Team member                     | work hours   |
|:--------------------------------| ------------:|
| Lucas Giordano                  |    63h       |
| Augustin Kapps                  |    56h       |
