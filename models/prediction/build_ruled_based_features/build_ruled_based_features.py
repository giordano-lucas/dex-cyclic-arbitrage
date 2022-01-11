import sys, os 
sys.path.append('/'.join(os.getcwd().split('/')[:4]))

from data_processing.build_embedding_features import run

base_config = {
        'use_liquid': True , 
        'nrows':10_000_000, 
        'drop_columns':None,
        'log_transformation':False,
        'new_train_idx':False,
        'extra_dir': 'ruled_based'
}
run(**base_config)

run(skip_split=True, features_name='encoded', **base_config)