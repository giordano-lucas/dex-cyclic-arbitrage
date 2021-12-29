## Abstract

Recent papers ([https://arxiv.org/abs/2105.02784](https://arxiv.org/abs/2105.02784)) showed a large number of occurrences of cyclic arbitrages in decentralised exchanges compared to centralised ones. In their work, they mainly focus on analysing these cycles in terms of length, token distribution, daily patens and profitability.

However, the factors driving their appearance have not been studied yet.  To this end, we propose to extend the work of [https://arxiv.org/abs/2105.02784](https://arxiv.org/abs/2105.02784) on Uniswap data. Moreover, we also plan to study the predictive power of these factors in a binary classification setting. It will allow determining whether or not a cycle can actually be implemented and generate a positive return, which has an inherent market value.

## Goal

The goal of this project is to study exploited cyclic arbitrage in decentralised exchanges. We already have access to a dataset containing cyclic arbitrages that were exploited. We intend to extract features out of events (trade rates, trade volumes, liquidity) preceding the arbitrages. 

These features could potentially be high dimensional (depending of the length of the time series) and we may need to create an embedding (through an auto encoder or a wavelet transform for instance) to build a relevant set of features of our future machine learning models.

Then, we will cluster the arbitrages based on the computed features. Ideally, we would like to observe meaningful clusterings: profitable cycles get clustered together, cycles having similar duration (how long it is profitable) also end up in the same cluster ... Once meaningful clusters are obtained, it gets interesting to use the same features in a prediction model having profitability of the arbitrage as a target.

## Data aquisition

### Data from the paper

We already have downloaded the dataset used in the arxiv paper. This dataset consists of arbitrage cycles that were exploited in the past. Each of these cycles is described by: a path (the token swaps),  a cost (gas fees), a profit etc. Let's call this dataset  "cycles_data"
### Uniswap data

A a second step, we construct a new dataset out of "cycles_data" as follows : for each cycle in "cycles_data" we fetch from [https://bitquery.io/](https://bitquery.io/) the "k" exchange rates preceding the swaps present in the cycle path.

## Test live plot


{% include_relative figures/Profitability_of_each_cluster_train.html %}