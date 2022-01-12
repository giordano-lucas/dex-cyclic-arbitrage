# Data Acquisition 
---

In this study, we propose to use the following datasets 
1. [Cyclic transaction dataset](https://disco.ethz.ch/misc/uniswap/cycles_in_Uniswap.json) : dataset made available in [Cyclic Arbitrage in Decentralized Exchange Markets](https://arxiv.org/abs/2105.02784). It contains information about arbitrage cycles that were exploited on DEXes. 
2. [Uniswap rates preceeding cyclic transaction dataset](https://www.kaggle.com/ogst68/uniswap-rates-preceeding-cyclic-arbitrages-raw/download) : dataset gathered in this study. It contains  the rates and gas prices preceeding cycles (600 transaction for each token pair uniswap pool).

To obtain these datasets, please follow the instruction below: 
1. Run the script : [download_uniswap_cycles.sh](scripts/data_processing.sh). It download the  `Cyclic transaction dataset`.
2. Download the `Uniswap rates preceeding cyclic transaction dataset` that was poseted on kaggle [here](https://www.kaggle.com/ogst68/uniswap-rates-preceeding-cyclic-arbitrages-raw/download) these data were previously fetched using the Bitquery platform (using the script `rates_from_Bitquery.py`). 

> **Note**: if you have access to the [IZAR EPFL cluster](https://www.epfl.ch/research/facilities/scitas/hardware/izar/), the simplest solution to get the datasets is to our `data` directory that was made available publicly under the following folder `scratch/izar/kapps/data/`.

## Notes

The [sanity check folder](sanity_checks) contains steps undertaken to compare the data fetched with data available on Etherscan to check the validity of our scripts.