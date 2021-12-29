## Abstract

Recent papers ([https://arxiv.org/abs/2105.02784](https://arxiv.org/abs/2105.02784)) showed a large number of occurrences of cyclic arbitrages in decentralised exchanges compared to centralised ones. In their work, they mainly focus on analysing these cycles in terms of length, token distribution, daily patens and profitability.

However, the factors driving their appearance have not been studied yet.  To this end, we propose to extend the work of [https://arxiv.org/abs/2105.02784](https://arxiv.org/abs/2105.02784) on Uniswap data. Moreover, we also plan to study the predictive power of these factors in a binary classification setting. It will allow determining whether or not a cycle can actually be implemented and generate a positive return, which has an inherent market value.

## Goal

The goal of this project is to study exploited cyclic arbitrage in decentralised exchanges. We already have access to a dataset containing cyclic arbitrages that were exploited. We intend to extract features out of events (trade rates, trade volumes, liquidity) preceding the arbitrages. 

These features could potentially be high dimensional (depending of the length of the time series) and we may need to create an embedding (through an auto encoder or a wavelet transform for instance) to build a relevant set of features of our future machine learning models.

Then, we will cluster the arbitrages based on the computed features. Ideally, we would like to observe meaningful clusterings: profitable cycles get clustered together, cycles having similar duration (how long it is profitable) also end up in the same cluster ... Once meaningful clusters are obtained, it gets interesting to use the same features in a prediction model having profitability of the arbitrage as a target.

## Data aquisition
Data used in the study come from two different sources: information about exploited cycles comes from the dataset used in the arxiv paper. Data concerning rates preceding the cycles come are from through bitquery.

### Data from the paper

We already have downloaded the dataset used in the arxiv paper. This dataset consists of arbitrage cycles that were exploited in the past. Each of these cycles is described by: a path (the token swaps),  a cost (gas fees), a profit etc.  It consists of a single JSON file and the downloading process is straightforward. Let's call this dataset  "cycles_data". ```cycles_data``` contains cycles of various lengths (number of tokens involved). The following figure displays the distribution of these lengths : 

Note that this figure is also present in the arxiv paper.
Moreover, to compute embeddings (next step) it would be more convenient to work on fixed-length cycles. Thus, cycles whose lengths are different than 3 are filtered out.  The obtained data is called ```filtered_cycles_data```.
While filtering, a new indexing system is created to identify cycles through an incremental  ```cycle_id```.
### Uniswap data
As a second step, we construct a new dataset out of "cycles_data" as follows: for each cycle in ```filtered_cycles_data``` we fetch from [https://bitquery.io/](https://bitquery.io/) the exchange rates and gas prices of the   ```k``` preceding the swaps present in the cycle path. This downloading process is more complex and needs to address some challenges. The free version of the bitquery API that is used for this project only allows a limited number of queries (10/mins). To solve this issue we used EPFL's cluster machine to query the API for a few weeks with a time delay between each call. To increase the throughput we used 2 API keys. For each given cycle in ```filtered_cycles_data``` 3 calls are needed, one per edge of the cycle. The fetched data are saved in multiple JSON files named ```uniswap_raw_data_start_end``` where ```start``` and ```end``` designate the starting end ending ```cycle_id``` of cycles included in the file. We chose to use multiple files to avoid big losses in case of failure, these files are copied in safe storage as a backup. The files are then combined into a single pandas DataFrame named  ```uniswap_data.csv```. Each row contains information about a single swap : ```cycle_id```, ```token1```, ```token2```, ```baseAmount```, ```quoteAmount```,```quotePrice```, ```gasPrice```, ```gasValue```, ```time```.

## Embedding computation

## Clustering

## Clustering validation

## Cycles predictability

## Test live plot


{% include_relative figures/Profitability_of_each_cluster_train.html %}
