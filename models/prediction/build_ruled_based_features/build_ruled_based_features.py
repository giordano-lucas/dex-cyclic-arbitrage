import sys, os 
sys.path.append('/'.join(os.getcwd().split('/')[:4]))

import data_processing.build_embedding_features as bef
import models.prediction.build_ruled_based_features.build_ruled_based_encoded_features as brbef
base_config = {
        'use_liquid': True , 
        'nrows':10_000_000, 
        'drop_columns':None,
        'log_transformation':False,
        'new_train_idx':False,
        'extra_dir': 'ruled_based'
}
brbef.run()
bef.run(skip_split=True, feature_name='encoded', **base_config, scaling=False)