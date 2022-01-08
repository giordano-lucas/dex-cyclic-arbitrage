import urllib.request
import sys 
import os
sys.path.append('/'.join(os.getcwd().split('/')[:4]))
from config.get import cfg

print('Start downloading')
url = 'https://disco.ethz.ch/misc/uniswap/cycles_in_Uniswap.json'
urllib.request.urlretrieve(url, cfg['directories']["data_dir"])