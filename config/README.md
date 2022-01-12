# Config 
---

This folder contains a configuration file : `config.yml` used throughout the entire project. 

It allows us to define directories and file names without having to specify the full path each time we want to use it. 

To import the config from other files of this repository, use the following syntax 

```python 
import sys, os 
sys.path.append('/'.join(os.getcwd().split('/')[:4]))

# import file config
from config.get import cfg
```

We also used a model config `models_config.yml` that contains the encoding layer in the different autoencoder's architecture proposed.