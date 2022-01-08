import json 
from config.get import cfg

def load_cycles(type_="raw",to_json=False):
    if type_== "raw":
        #"/scratch/izar/kapps/DEX-Cyclic-Arbitrage/data/cycles_in_Uniswap.json"
        data_path = cfg['files']['cycles_in_Uniswap']
    if type_== "filtered":
        data_path =  cfg['files']['filtered_cycles']
        #"/scratch/izar/kapps/DEX-Cyclic-Arbitrage/data/filtered_cycles.json"
    data = []
    for line in open(data_path):
        if to_json:
            data.append(json.loads(line))
        else :
            data.append(line)
    return data

