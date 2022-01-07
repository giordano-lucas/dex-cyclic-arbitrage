# DEX-Cyclic-Arbitrage



## data_acquisition
To obtain all the data needed for the analysis one shoud first run the python script
* `download_uniswap_cycles.sh` : this script download the  `cycles_in_Uniswap.json` dataset published with the arxiv paper `Cyclic Arbitrage in Decentralized Exchange Markets`. It contains information about arbitrage cycles that were exploited on DEX 
Then, to download the rates and gas prices preceeding cycles there exists 2 possibilities :
* Method 1: Use `rates_from_Bitquery.py` to query the data from Bitquery. Because of queying constraints, this solution is slow and will take weeks to get the data
* Method 2: Download the data that we poseted on kaggle [here](https://www.kaggle.com/ogst68/uniswap-rates-preceeding-cyclic-arbitrages-raw/download) these data were previously fetched using method1. 

## data_processing
To process the downloaded data on need to follow these steps :
    1. run `filter_cycles.py` to filter cycles of  `cycles_in_Uniswap.json` on their lenghts. Give the desired lenght in argument (default is 6). The script produces a new dataset : `data/filtered_cycles.json`
    2. run `combine_raw_data.py` to combine the multiple fetched files available at `data/uniswap_raw_data/`. It produces a single .csv file (`data/uniswap_data.csv`)containing all rates and gas prices that were queried from bitQuery.   
    3. run `build_embedding_features.py` to create, scale and pad a train (`data/ML_features/raw_train_features.npy`) and test (`data/ML_features/raw_test_features.npy`) set out of `data/uniswap_data.csv`. These set a meant to be used to train an autoencoder for feature extraction.
    4. run `compute_embeddings.py` using the model we trained or yours (see models section) in argument to compute the embeddings of the previously generated train and test sets. The execution produces 2 new datasets : `data/ML_features/encoded_train_features.npy` and `data/ML_features/encoded_test_features.npy`
    5. run `build_prediction_data.py` to prepare the data needed for the prediction task. It produces train and test sets containing the profitability ofeach cycles as well as the tokens involved in the cycles.     
## models