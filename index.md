# Abstract

The paper Cyclic Arbitrage in Decentralized Exchange Markets[[1]](https://arxiv.org/abs/2105.02784) showed a large number of occurrences of cyclic arbitrages in decentralised exchanges compared to centralised ones. In their work, they mainly focus on analysing these cycles in terms of length, token distribution, daily patens and profitability.

However, the factors driving their appearance have not been studied yet.  To this end, we propose to extend the work of [[1]](https://arxiv.org/abs/2105.02784) on Uniswap data. Moreover, we also plan to study the predictive power of these factors in a binary classification setting. It will allow determining whether or not a cycle can actually be implemented and generate a positive return, which has an inherent market value.

# Goal

The goal of this project is to study exploited cyclic arbitrage in decentralised exchanges. We already have access to the [Cyclic transaction dataset](https://disco.ethz.ch/misc/uniswap/cycles_in_Uniswap.json) which contains cyclic arbitrages that were exploited. We intend to extract features out of events (trade rates, trade volumes, liquidity) preceding the arbitrages. 

These features could potentially be high dimensional (depending of the length of the time series) and we will need to use dimensionality reduction techinques to create an embedding to build a relevant set of features of our future machine learning models.

Then, we will cluster the arbitrages based on the computed features. Ideally, we would like to observe meaningful clusterings: profitable cycles get clustered together, cycles having similar duration (how long it is profitable) also end up in the same cluster, etc. Once meaningful clusters are obtained, it gets interesting to use the same features in a prediction model having profitability of the arbitrage as a target.

# Data aquisition

Data used in the study come from two different sources: information about exploited cycles comes from the dataset used in the arxiv paper. Data concerning rates preceding the cycles come are from through bitquery.

## Dataset from the paper

We already have downloaded the dataset used in the arxiv paper. This dataset consists of arbitrage cycles that were exploited in the past. Each of these cycles is described by: a path (the token swaps),  a cost (gas fees), a profit etc.  It consists of a single JSON file and the downloading process is straightforward. The `cyclic transaction dataset` contains cycles of various lengths (number of tokens involved). The following figure displays the distribution of these lengths : 

XXXXXXX

> Note : this figure was taken from the arxiv paper.

Moreover, to compute embeddings (next step) it would be more convenient to work on fixed-length cycles. Thus, cycles whose lengths are different than 3 are filtered out. The obtained data is called `filtered_cycles_data`.
While filtering, a new indexing system is created to identify cycles through an incremental  `cycle_id`.

## Custom extented dataset (Uniswap data)

As a second step, we construct an extented version of the `cyclic transaction dataset` as follows. For each cycle in `filtered_cycles_data`: 

1. we fetch from [https://bitquery.io/](https://bitquery.io/) the exchange rates and gas prices of the   `k` preceding the swaps present in the cycle path. This downloading process is more complex and needs to address some challenges. The free version of the bitquery API that is used for this project only allows a limited number of queries (10/mins). To solve this issue we used EPFL's cluster machine to query the API for a few weeks with a time delay between each call. To increase the throughput we used 2 API keys. For each given cycle in `filtered_cycles_data` 3 calls are needed, one per edge of the cycle. 
2. The fetched data is saved in multiple JSON files named `uniswap_raw_data_start_end` where `start` and `end` designate the starting end ending `cycle_id` of cycles included in the file. We chose to use multiple files to avoid big losses in case of failure, these files are copied in safe storage as a backup. The files are then combined into a single pandas DataFrame named  `uniswap_data.csv`. 

Each row contains information about a single swap :

>  `cycle_id`, `token1`, `token2`, `baseAmount`, `quoteAmount`,`quotePrice`, `gasPrice`, `gasValue`, `time`.

In the following section, we will only work on this extented dataset which is therefore refered to as `the dataset`. 

# Replication of some of [[1]](https://arxiv.org/abs/2105.02784) figures

In order to introduce the `Cyclic transaction dataset` and check our understanding of the data, we propose to repropose a few figures of the arxiv paper.

# Data Wrangling 
## Data Exploration

Before developping any machine learning model, we need to grap a basic understanding of the `dataset` and its statistical properties. 

Since we are dealing with financial time series for the following features:
>  `baseAmount`, `quoteAmount`,`quotePrice`, `gasPrice`, `gasValue`
 we will probably observe heavy tail distribution to some extent. It is what we are going to check first.

> Note : in this section, no distinction is made between cycles. In other words, all data available is treated as a single feature, aka a single distribution. Indeed, understanding each cycle separatly is a cumbersome process. Furthermore, it does not help in getting a global understanding of the dataset.


In this first milestone, only `quotePrice` and `gasPrice` are used as features for the embedding (in fact `quotePrice = quoteAmount/baseAmount` ). Therefore, we propose the following plots:

{% include_relative figures/data_exploration/quotePrice.html %}

{% include_relative figures/data_exploration/gasPrice.html %}

## Further preprocessing

As it was previously said, the focus of this study is on cycles of length 3. 

In this first step, each cycle is modelled as a list of 3 nodes which are themselves modelled as 2-dimensional time-series : prices and gas. 


    - ThMean gas price over the spanned period
    - Liquidity of the pools

The spanned period has to be defined and backtested (e.g. fixed or mean frequency between two cycles).

### Zero-Padding

### building the ML feature tensor
### taking the log
### train test split
### Scaling


# Cycle embedding 

In section [Further preprocessing](#further-preprocessing), we modelled each cycle by a `D x P x K = 3 x 600 x 2` tensor. Hence 

To create the embedding, we propose the following approaches:

- Use convolutional auto-encoders to create the embedding
- Compare performance with Fourier/Wavelet transform or Signature transform.

The task of the autencoder is summarised in the following figure.

<p align="center">
<img width="400" alt="Diagram encoder" src="figures/diagrams/encoder/encoder-diagram.drawio.svg">
</p>

Through this process, we observe a reduction factor of `3600/100 = 36x` which is non negligeable. 


In the classification task, we will be able to measure the gain of the embedding compared to the raw features (base model).

# Clustering

## Motivation and method

Run different clustering algorithms (k-means or db-scan).

For each of them, we plan the validity of the clustering through the following metrics

- Same sized cycles are in the same cluster.
- Similar cycles (in terms of common nodes) are in the same cluster.
- Cycles with similar profitability (yield, positivity) are clustered together.

This step will allow us to quantify how much information (and possibly predictive power) is contained in the clustering.

When the validity of the clustering is established, we can start to analyse it. We propose to study the following list of factors (could be extended or reduced later on): Gas, Time, Market cap, volume, liquidity, volatility. To this end, we propose to compute several metrics for each cluster, observe if they are any differences across clusters.

## Clustering validation

# Cycles predictability

# Test live plot


{% include_relative figures/clustering/Profitability_of_each_cluster_train.html %}

# Further steps

## Embedding improvement

In section [Further preprocessing](#further-preprocessing), `0-padding` was introducted to standardize the length of each time series. However, the choice of `0s` is rather arbitrary and can introduce many problems upon training the autoencoder (as well as scaling the data). Indeed, a small computation shows that introducing this padding technique adds XXX zeros which corresponds to a fraction XXX % of the training set entries. This means that the autoencoder can do a decent only by trying to improve the reconstruction of `0s`is the training set. 

Moreover, if we keep increasing the number of padded `Os`, we can make the `MSE` abitrary close to 0 (perfect model). 

These undesired behaviors could be addresssed by introducing a special token `PAD` which has no intrinsic value (similarly as in the `BERT Transformer` model). Defining a custom `keras model` and `MSE` that simply ignore these `PAD` tokens would increase the quality of the embedding.

Another possible improvement is to define a custom convolutional layer that can take advantage of the cyclic nature of the arbitrage. Circular convolution is be a step towards this direction.


Finally, we should include other features (e.g. the 3 encodings of the tokens present in the cycle) to the model. However, this process is not as straightforward as it may seem. Indeed  




## DBscan 