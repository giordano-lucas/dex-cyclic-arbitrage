import json 

def load_cycles(type_="raw",to_json=False):
    if type_== "raw":
        data_path = "/scratch/izar/kapps/DEX-Cyclic-Arbitrage/data/cycles_in_Uniswap.json"
    if type_== "filtered":
        data_path = "/scratch/izar/kapps/DEX-Cyclic-Arbitrage/data/filtered_cycles.json"
    data = []
    for line in open(data_path):
        if to_json:
            data.append(json.loads(line))
        else :
            data.append(line)
    return data

