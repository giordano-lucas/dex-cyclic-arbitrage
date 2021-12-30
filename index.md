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

<p align="center">
<img width="400" alt="cycles length distribution" src="figures/data_exploration/cycles_length_distribution.html">
</p>

> Note: this figure was taken from the arxiv paper.

Moreover, to compute embeddings (next step) it would be more convenient to work on fixed-length cycles. Thus, cycles whose lengths are different than 3 are filtered out. The obtained data is called `filtered_cycles_data`.
While filtering, a new indexing system is created to identify cycles through an incremental  `cycle_id`.

## Custom extented dataset (Uniswap data)

As a second step, we construct an extented version of the `cyclic transaction dataset` as follows. For each cycle in `filtered_cycles_data`: 

1. We fetch from [https://bitquery.io/](https://bitquery.io/) the exchange rates and gas prices of the   `k` preceding the swaps present in the cycle path. This downloading process is more complex and needs to address some challenges. The free version of the `bitquery API` that is used for this project only allows a limited number of queries (10/mins). To solve this issue we used EPFL's cluster machine to query the API for a few weeks with a time delay between each call. To increase the throughput we used 2 API keys. For each given cycle in `filtered_cycles_data` 3 calls are needed, one per edge of the cycle. 
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

> Note: in this section, no distinction is made between cycles. In other words, all data available is treated as a single feature, aka a single distribution. Indeed, understanding each cycle separatly is a cumbersome process. Furthermore, it does not help in getting a global understanding of the dataset.

In this first milestone, only `quotePrice` and `gasPrice` are used as features for the embedding (in fact `quotePrice = quoteAmount/baseAmount` ). Therefore, we propose the following plots:

{% include_relative figures/data_exploration/quotePrice_small.html %}

{% include_relative figures/data_exploration/gasPrice_small.html %}

We observe that both features are extremely heavy tailed. It is likely to cause some issues whem used as features for machine learning models. As shown in the plot,applying a logarithmic transformation make the distributions more Gaussian (desired behavior). 

## Data preprocessing

As it was previously said, the focus of this study is on cycles of length 3. 

In this first milestone, we propose to model each cycle as a list of 3 nodes which are themselves represented as 2-dimensional time-series: `quote prices` and `gas prices`. These time series have length at most `P = 600` (data fetched from `bituery`).

### Logarithmic transformation

As shown in section [Data Exploration](#data-exploration), it is probably a good idea to apply the `element-wise logarithm` to the `quotePrice` and `gasPrice` features as a first step.
### Zero-Padding

In this study, each cycle is represented as a tensor with the following 3 dimentions :

| Name | Size |  Description                                                           |
|:----:|-----:|------------------------------------------------------------------------|
| `D`  | 3    | the length of the cycle, aka the number of tokens                      |
| `P`  | 600  | the length of the time series of swap transactions                     |
| `K`  | 2    | the number of time series/features (`quotePrice` and `gasPrice`)       |

However, due to lack of liquity in Uniswap pools for a given pair of tokens, there could be less than `P = 600` swaps transaction fetched from `bitquery`.  Since machine learning models required fixed input size for their functionning, we need to pad the shorter time series. To this end, the fixed length `P =600` was chosen, as well as the `zero-padding` techinque for simplicity reasons.

>  Note: the spanned period was fixed to be `P = 600` is this first milestone. It will be backtested (e.g. fixed or mean frequency between two cycles) later on.

### Building the feature tensor

`The dataset` is not directly shaped to build the tensor feature for the machine learning models. Indeed it simply contains a list of swap transactions in `csv` format. We need to massage using with multiple operations to group transactions associated to the same cycle together and pad each time series independently. 

> Note: hardware capacities of the cluster only allowed us to process `25 000 000` swap transactions at once. We could, later on, consider batch processing to handle more transactions. 

### Train/Test split

In order to test our models with the least amount of bias possible,we randomly split the datasets into a training and testing set (`30%`). 
### Scaling the features

To avoid the effect of the scales of the features in our results we have to re-scale them. To do so, we can consider two different approaches:

* Use a standard scaler that applies the following transformation to each feature $x_i$ :
        $$x_i=\frac{x_i-\bar{x_i}}{std(x_i)}$$
  By doing so we obtain a dataset with zero mean and a variance of 1 
* Use a min max scaler that reduce the range fo the features to [0,1] by applying the following transformation to each feature $x_i$ :
         $$x_i=\frac{x_i-min(x_i)}{max(x_i)-min(x_i)}$$

After taken the `log` features somhow look Gaussian, so we decided to opt for the standard scaler in this first milestone.
# Cycle embedding 

## Movitation

In section [Data preprocessing](#data-preprocessing), we modelled each cycle by a `D x P x K = 3 x 600 x 2` tensor. Hence, each of them is represented by `3600` parameters. 

However, one could argue that `3600` is  a too high dimensional way of encoding cycles. Espectially given the fact we are dealing with financial time series containing lots of noise. Building a latent representation of cycles in which distances are more effectively modelled which can be a great use in futher studies. 

In fact, later on in this article, we are going to cluster cycles using the `KMeans` algortihm which heavily relies on meaningfull euclidian distances computation which can make great use of a dimensionality reduction.

To create this embedding, multiple approaches can be considered. We propose the following:

1. [Principal component analysis (PCA)](https://en.wikipedia.org/wiki/Principal_component_analysis) 
2. [Autoencoder](https://en.wikipedia.org/wiki/Autoencoder) 

`PCA` is refered to in the litterature as a simple linear projection operation. We plan to use it as a baseline comparison with the more complex `autoencoder` model.

The task of an autencoder is summarised in the following figure.

<p align="center">
<img width="400" alt="Diagram encoder" src="figures/diagrams/encoder/encoder-diagram.drawio.svg">
</p>

> Note: through this process, we observe a reduction factor of `3600/100 = 36x` which is non-negligeable.

To better capture the structure of cycles, a convolutional autoencoder will be used to create the embedding. The idea is that when a cyclic arbitrage is implemented, the first transaction could affect some the price/gas of the second token and similarly for other transactions. The convolution operations could extract these neighborings relationships between tokens in order to build a better latent representation of cycles.

In the [Cycles profitability prediction](#cycles-profitability-prediction) task, we will be able to measure the gain of the embedding compared to the raw features (base model).

## Performance




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

{% include_relative figures/clustering/Profitability_of_each_cluster_train.html %}
# Cycles profitability prediction

# Further steps/improvements
## Embedding improvement

In section [Data preprocessing](#data-preprocessing), `0-padding` was introducted to standardize the length of each time series. However, the choice of `0s` is rather arbitrary and can introduce many problems upon training the autoencoder (as well as scaling the data). Indeed, a small computation shows that introducing this padding technique adds XXX zeros which corresponds to a fraction XXX % of the training set entries. This means that the autoencoder can do a decent only by trying to improve the reconstruction of `0s`is the training set. 

Moreover, if we keep increasing the number of padded `Os`, we can make the `MSE` abitrary close to 0 (perfect model). 

These undesired behaviors could be addresssed by introducing a special token `PAD` which has no intrinsic value (similarly as in the `BERT Transformer` model). Defining a custom `keras model` and `MSE` that simply ignore these `PAD` tokens would increase the quality of the embedding.

Another possible improvement is to define a custom convolutional layer that can take advantage of the cyclic nature of the arbitrage. Circular convolution is be a step towards this direction.

Finally, we should include other features (e.g. the 3 encodings of the tokens present in the cycle) to the model. However, this process is not as straightforward as it may seem. Indeed XXXXX

If we have time, we could also compare performance of the encoder with Fourier/Wavelet transform or Signature transform.

## Clustering improvement 

If we increase the quality of the embedding, the clustering quality should increase as well. However, we can futher backtest the clustering algorthm by comparing KMeans with BDscan for instance.
