# go in the good directory
cd data_processing
# create filter_cycles dataset
python3 filter_cycles.py
# merge cycles with uniswap dataset
python3 combine_raw_data.py
# build tensor for embedding
python3 build_embedding_features.py
# build prediction data
python3 build_prediction_data.py