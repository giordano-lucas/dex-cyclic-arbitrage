# Rule-based embedding
---

## Movitation

To evaluate the performance of our cycle embedding (autoencoder), we propose to study the impact of different embeddings features on the output of a binary classification task, precisely the profitability of a cycle.

The emphasis is not put on finding the best overall model here. The idea is to study the difference in the confusion matrix and metrics (accuracy, f1-score, recall, precision) that occur when the input features change.

If we observe better performance metrics for our autoencoder embedding, this experiment will have provided evidence that our embedding somehow captures the underlying structure of cyclic arbitrages. 

## Method 

### Embeddings used

To this end, we propose to study 3 different types of embedding :

1. The AE embedding (base hypothesis)
2. The PCA embedding (alternative hypothesis 1)
3. A rule-based embedding using technical indicators (alternative hypothesis 2)

If the first two options are described in detail in the earlier steps of this project, it is not the case for the third one. 

The following rolling indicators are used :

1. SMA
5. Rolling volatility

using two different rolling windows (`5` and `20`).

These indicators are applied on the following underlying time-series data (for each cycle):

1. `Quote price` 
2. `Gas price`
3. Log-returns of `Quote price`
4. Log-returns of `Gas price`

In other words, for each cycle, we construct `4 * 2 * 2 = 16` features.

After `zero-padding` the tensor build is of shape `N x 3 x 600 x 16`.

> **Note**: the `NaN` introduced by the computed are filled using `0` to have comparable shapes with the AE features.

Again since we have a massive imbalance between classes the `accuracy` metrics needs to be avoided. We will investigate the differences in terms of `precision` and `recall`, in particular through the `f1-score` metric.


### Performance analysis

Again since we have a massive imbalance between classes the `accuracy` metrics needs to be avoided. We will investigate the differences if terms of `precision` and `recall`, in particular through the `f1-score` metric.

## Notes to the reader

### Comments on implementation

The construction of these indicators was rather cumbersome. Indeed the size of data set of features used to train the autoencoder was chosen according to the hardware capabilities of the cluster (`10 000 000` rows). However, for the latter two features were used (`GasPrice` and `QuotePrice`) but for the rule-based dataset, we have not less than 12 features ! In other words, the size of the dataset is more or less multiplied by 6. 

We tried many different implementations to make it run of the cluster without our process being killed. The implementation is this folder is the most efficient we could come up with and it used Python's garbage collector to manually free the memory of unused variables.

> **Note**: we tried to use `pandas-ta` library (`ta.Strategy` and others) to ease the development process. However, it was too slow and memory intense on a Grouper object. Hence, we had to use a pure `pandas` based implementation.

In fact, the number of indicators/features used was chosen to be 12 to make it possible to run on the cluster. It is not possible to another one. 

### Data preprocessing

In terms of data, the rule-based dataset is available under the following names in the `config.yml` file:

- `encoded_train_features` : from the raw ae training set, the technical indicators are built.
- `encoded_test_features` :  from the raw ae test set, the technical indicators are built.
- `scaled_encoded_train_features` : tensor padded version of `encoded_train_features`.
- `scaled_encoded_test_features`  : tensor padded version of `encoded_test_features`.

In order to generate then use the following command:

```shell
python3 models/prediction/build_rule_based_features/build_rule_based_scaled_encoded_features.py
```


  