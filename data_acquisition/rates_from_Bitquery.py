import json
import requests
import time 
import gzip
import sys 
sys.path.append('/'.join(os.getcwd().split('/')[:4]))
from helper import *
from config.get import cfg

API_KEY1 = 'BQYd6qf95gjz6MU3FTP9yQegkGVkuQ4r' # Augustin's KEY
API_KEY2 = 'BQYt1q57Typ8HsY7nxzqsGzmKA0u7FzS' # Lucas's KEY
print("===============================================")

print("Loading filtered cycles...")
filtered_cycles = load_cycles("filtered")

print(len(filtered_cycles)," cycles loaded")
print("extracting paths")
cycle_paths = []
for cycle in filtered_cycles:
    cycle = json.loads(cycle)
    
    block_n  = max([x["blockNumber"] for x in cycle["receipt"]["logs"]])
    block_n  = int(block_n,16) 
    cycle_id = cycle["cycle_id"]
    path     = cycle["path"]
    
    cycle_paths.append({"block_n":block_n,"cycle_id":cycle_id,"path":path})

def run_query(query,API_KEY):  # A simple function to use requests.post to make the API call.
    headers = {'X-API-KEY': API_KEY }
    request = requests.post('https://graphql.bitquery.io/',
                            json={'query': query}, headers=headers)
    if request.status_code == 200:
        return request.json()
    else:
        raise Exception('Query failed and return code is {}.      {}'.format(request.status_code,
                        query))
        
def create_query(k,token1,token2,block_n):
    return ("""{
  ethereum(network: ethereum) {
    dexTrades(
      options: {limit: """+str(k)+""", asc: "timeInterval.second"}
      exchangeName: {is: "Uniswap"}
      baseCurrency: {is: \""""+token1+"""\"}
      quoteCurrency: {is: \""""+token2+"""\"}
      height: {between: [0,"""+str(block_n)+"""]}
    ) {
      timeInterval {
        second(count: 1)
      }
      baseCurrency {
        symbol
        address
      }
      baseAmount
      quoteCurrency {
        symbol
        address
      }
      gasPrice
      gasValue
      transaction{
        hash
      }
      quoteAmount
      trades: count
      quotePrice
      maximum_price: quotePrice(calculate: maximum)
      minimum_price: quotePrice(calculate: minimum)
      open_price: minimum(of: block, get: quote_price)
      close_price: maximum(of: block, get: quote_price)
    }
  }
}""")


print("start fetching data")

k = 600 # number of data point to fetch per request
first_cycle  = 80000
last_cycle   = 120000

uniswap_raw_data_path = cfg['directories']['uniswap_raw_data']+f"/uniswap_raw_data_{first_cycle}_{last_cycle}.json.gz"
f_out = gzip.open(uniswap_raw_data_path,"wt")
n_query   = 0
err_count = 0
start_time = time.time()
print(f"start fetching, from {first_cycle} to {last_cycle}")
for cycle_path in cycle_paths[first_cycle:last_cycle]:
    elapsed = time.time() -start_time
    block_n = cycle_path["block_n"]
    cycle_id = cycle_path["cycle_id"]
    path = cycle_path["path"]
    for i in range(0,len(path),2):
        if n_query>0 and n_query/2 %10==0: # /2 comes from the fact that we have 2 API keys
            time.sleep(60 - elapsed%60 + 7) # wait the time needed to reach next minute + 7s
        
        pair = (path[i:i+2])
        fetched_data = {}
        fetched_data["pair"] = pair
        fetched_data["cycle_id"] = cycle_id
        query = create_query(k,token1=pair[0]
                                     ,token2=pair[1],
                                     block_n=str(block_n))
        try : 
            if n_query%2==0:
                fetched_data["data"] = run_query(query,API_KEY1)
            else:
                fetched_data["data"] = run_query(query,API_KEY2)

            n_query+=1
            f_out.write(json.dumps(fetched_data)+"\n")
            print(f"{n_query} querries, {err_count} errors, progression :{100*(cycle_id-first_cycle)/(last_cycle-first_cycle):0.4f}% , {elapsed:0.0f}s ",end="\r")
        except :
            err_count+=1
        
print()
f_out.close()
print(f"DONE")