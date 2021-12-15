import csv
import gzip
import sys 
sys.path.append("/scratch/izar/kapps/DEX-Cyclic-Arbitrage/")
from helper import *

import glob 

from config.get import cfg

print("===========================================")
print("Loading uni data")

uni_raw_data_paths = glob.glob(cfg['directories']['uniswap_raw_data'])

uniswap_raw_data = []
for path in uni_raw_data_paths:
    f_in = gzip.open(path,"rt")
    for line in  f_in:
        uniswap_raw_data.append(line)
    f_in.close()

print(f"{len(uniswap_raw_data)} data loaded")   
def load_transaction_hashs():
    """load the cycles transaction hashs in a dictionary with cycle_id as key, only works on the filtered cycles dataset"""
    filtered_cycles = load_cycles("filtered")
    res = {}
    for cycle in filtered_cycles:
        cycle = json.loads(cycle)
        cycle_id = cycle.pop("cycle_id")
        data = cycle
        res[cycle_id] = data["receipt"]["logs"][0]['transactionHash']
    return res


print("Loading tx hashs")
cycle_transaction_hashs = load_transaction_hashs()
print("processing")
uni_data_path = "/scratch/izar/kapps/DEX-Cyclic-Arbitrage/data/uniswap_data.csv"
with open(uni_data_path, mode='w') as f:
    csv_writer = csv.writer(f, delimiter=',')
    header = ['cycle_id', 'token1', 'token2','baseAmount',"quoteAmount","quotePrice","gasPrice","gasValue","time"]
    csv_writer.writerow(header)
    for data in uniswap_raw_data:
        data = json.loads(data)
        cycle_id = data["cycle_id"]
        
        trades = data["data"]["data"]["ethereum"]["dexTrades"]
        if not trades:
            continue
        tx_hash_cycle = cycle_transaction_hashs[cycle_id]
        for trade in trades:
            tx_hash_trade =  trade['transaction']["hash"]
            if tx_hash_trade==tx_hash_cycle:
                # we do not want to consider trades happening after the cycle
                break
            token1 = trade['baseCurrency']['symbol']   
            token2 = trade['quoteCurrency']['symbol'] 
            baseAmount  = trade['baseAmount']
            quoteAmount = trade['quoteAmount']
            quotePrice  = trade['quotePrice']
            gasPrice    = trade['gasPrice']
            gasValue    = trade['gasValue']
            time        = trade['timeInterval']['second']
            csv_writer.writerow([cycle_id, token1, token2,baseAmount,quoteAmount,quotePrice,gasPrice,gasValue,time])