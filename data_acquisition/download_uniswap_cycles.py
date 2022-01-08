import urllib.request
import sys 
import os
sys.path.append('/'.join(os.getcwd().split('/')[:4]))
from config.get import cfg
from helper import check_and_create_dir

check_and_create_dir(cfg['directories']["data_dir"])
print('Start downloading')
url = 'https://disco.ethz.ch/misc/uniswap/cycles_in_Uniswap.json'
urllib.request.urlretrieve(url, cfg['files']["all_cycles"])