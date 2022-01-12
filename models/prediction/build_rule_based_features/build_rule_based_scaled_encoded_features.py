import sys, os 
sys.path.append('/'.join(os.getcwd().split('/')[:4]))

import data_processing.build_embedding_features as bef
import models.prediction.build_rule_based_features.build_rule_based_encoded_features.py as brbef

base_config = {
        'use_liquid': True , 
        'nrows':10_000_000, 
        'drop_columns':None,
        'log_transformation':False,
        'new_train_idx':False,
        'extra_dir': 'ruled_based',
        'skip_split': True,
        'feature_name': 'encoded',
        'scaling':False,
}
# construct encoded train/test sets
brbef.run()
# scaling and tensor building
bef.run(**base_config)