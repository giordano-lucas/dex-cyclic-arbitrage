# Data Processing
---

To process the downloaded data one need to follow these steps :
    1. run `filter_cycles.py` to filter cycles of  `cycles_in_Uniswap.json` on their lenghts. Give the desired lenght in argument (default is 6). The script produces a new dataset : `data/filtered_cycles.json`
    2. run `combine_raw_data.py` to combine the multiple fetched files available at `data/uniswap_raw_data/`. It produces a single .csv file (`data/uniswap_data.csv`)containing all rates and gas prices that were queried from bitQuery.   
    3. run `build_embedding_features.py` to create, scale and pad a train (`data/ML_features/raw_train_features.npy`) and test (`data/ML_features/raw_test_features.npy`) set out of `data/uniswap_data.csv`. These set a meant to be used to train an autoencoder for feature extraction.
    4. run `compute_embeddings.py` using the model we trained or yours (see models section) in argument to compute the embeddings of the previously generated train and test sets. The execution produces 2 new datasets : `data/ML_features/encoded_train_features.npy` and `data/ML_features/encoded_test_features.npy`
    5. run `build_prediction_data.py` to prepare the data needed for the prediction task. It produces train and test sets containing the profitability ofeach cycles as well as the tokens involved in the cycles.    

> **Note**: to simply the the data processing procedure for the reader, we created a shell script : [data_processing.sh](scripts/data_processing.sh). In the base directory, run the following command:

```bash
bash scripts/data_processing.sh
```