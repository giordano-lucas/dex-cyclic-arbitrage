import csv
import gzip
import sys 
sys.path.append("/scratch/izar/kapps/DEX-Cyclic-Arbitrage/")
from helper import *

import glob 

from config.get import cfg


 
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

uni_raw_data_paths = glob.glob(cfg['directories']['uniswap_raw_data']+"*json.gz")

print("Loading tx hashs")
cycle_transaction_hashs = load_transaction_hashs()
print("processing")
#"/scratch/izar/kapps/DEX-Cyclic-Arbitrage/data/uniswap_data.csv"
uni_data_path = cfg["files"]["preprocessed_data"] 
count = 0
with open(uni_data_path, mode='w') as f_out:
    csv_writer = csv.writer(f_out, delimiter=',')
    header = ['cycle_id', 'token1', 'token2','baseAmount',"quoteAmount","quotePrice","gasPrice","gasValue","time"]
    csv_writer.writerow(header)   
    for path in uni_raw_data_paths:
        print(path)
        f_in = gzip.open(path,"rt")
        for data in f_in:
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
                count+=1
                if count%10_000==0:
                    print(count,end="\r")
        f_in.close()