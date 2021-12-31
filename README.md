<img width="1665" alt="Screenshot 2021-12-29 at 09 52 48" src="https://user-images.githubusercontent.com/43466781/147819628-ed23274e-8d05-487d-b499-2b7a51e36eee.png">

## Abstract

XXXXXXXXX

## Research questions

XXXXXXXXX

## Proposed datasets

1. [Cyclic transaction dataset](https://disco.ethz.ch/misc/uniswap/cycles_in_Uniswap.json)
2. [Uniswap rates 1.0](https://noting)


## Methods :

XXXXXXXXXXXX

## Timeline and contributions :

### Week 1 : Data Acquisition & Setup (1)

| Task                                                                    | Team member(s)                  | work hours  |
| :-----------------------------------------------------------------------|:--------------------------------| -----------:|
| Literature research                                                     | Lucas & Augustin                | 3h          |
| API choice and query design                                             | Lucas & Augustin                | 4h          |
| EPFL Cluster & environment setup                                        | Lucas                           | 2h          |
| Data fetching script test                                               | Augustin                        | 3h          |
| Data fetching validation                                                | Augustin                        | 1h          |
| Data fetching improvements                                              | Augustin                        | 2h          |


### Week 2 : Data preprocessing

| Task                                     | Team member(s)                  | work hours  |
| :----------------------------------------|:--------------------------------| -----------:|
| Data cleaning                            | Augustin                        | 4h          |
| Data exploration                         | Lucas                           | 1h          |
| Raw data => embedding format             | Lucas                           | 3h          |


### Week 3 : Embedding & Clustering 

| Task                                     | Team member(s)                  | work hours  |
| :----------------------------------------|:--------------------------------| -----------:|
| Autencoder keras basic code              | Lucas                           | 2h          |
| Comparision with PCA and debugging       | Lucas                           | 1h          |
| K-means                                  | Augustin                        | 1h          |

### Week 4 : Clustering analysis, Profitablity prediction & report writing

| Task                                    | Team member(s)                  | work hours  |
| :---------------------------------------|:--------------------------------| -----------:|
| Clustering analysis                     | Lucas                           | 4h          |
| Profitablity prediction setup           | Augustin                        | 2h          |
| Github pages setup                      | Lucas                           | 2h          |
| Data story (1)                          | Lucas                           | 5h          |
| Data story (2)                          | Augustin                        | 2h          |

### Week 5 : Hyperparameter opmisation and improvements & report writing

| Task                                    | Team member(s)                  | work hours  |
| :---------------------------------------|:--------------------------------| -----------:|
| Hyperparameter opmisation               | Lucas & Augustin                |           |
| Notebook comments and markdown          | Lucas & Augustin                |           |
| Data story (2)                          | Lucas & Augustin                |           |

### Total contribution:

| Team member                     | work hours   |
|:--------------------------------| ------------:|
| Lucas Giordano                  |           |
| Augustin Kapps                  |           |


## Notes to the reader
### Organisation of the repository

    .
    ├── data                                      # Data folder
    │ ├── XXXXXXXXXXXX                      '     # XXXXXXX
    │ ├── XXX                                     # folder containing XXX
    │ │  ├── XXXXX.csv                            # example of file
    │ │  ├── ...
    │ ├── XXXXXX                                  # folder containing XXXX
    │ │  ├── XXXX                                 # contains the data to be loaded by geopandas
    │ │  │  ├── XXXXXXXXXXXXXX                    # example of file
    │ │  │  ├── ...
    │ │  ├── ...
    ├── figures                             # Contains the ouput images and html used for the data story
    ├── requirements.txt                    # Dependencies
    └── README.md               
    

### Dependencies requirement

In the repository, we provide a `requirement.txt` file from which you can create a virutatal python environment.

> Note : to be able to import `talos` on the Scitas cluster, we need to update line 8 of `opt/venv-gcc/lib/python3.7/site-packages/kerasplotlib/traininglog.py` from `from keras.callbacks import Callback` to `from tensorflow.keras.callbacks import Callback`
