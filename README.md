# Anomalous time-series

    UNFINISHED TRANSLATION

Source: [robjhyndman/anomalous](https://github.com/robjhyndman/anomalous) R package

> It is becoming increasingly common for organizations to collect very large amounts of
 data over time, and to need to detect unusual or anomalous time series. 
 For example, Yahoo has banks of mail servers that are monitored over time. 
 Many measurements on server performance are collected every hour for each of thousands 
 of servers. A common use-case is to identify servers that are behaving unusually. 
 Methods in this package compute a vector of features on each time series, measuring 
 characteristics of the series. For example, the features may include lag correlation,
 strength of seasonality, spectral entropy, etc. Then a robust principal component 
 decomposition is used on the features, and various bivariate outlier detection 
 methods are applied to the first two principal components. This enables the most 
 unusual series, based on their feature vectors, to be identified. The bivariate outlier
 detection methods used are based on highest density regions and alpha-hulls. 
 For demo purposes, this package contains both synthetic and real data from Yahoo.

Simple Example:

```python
import pandas as pd
import numpy as np
import anomalous

# Create synthetic data
index = pd.date_range(start='2000', periods=100, freq="D")
b = []
for i in range(100):
    b.append(pd.Series(np.random.rand(100), index=index, name="var_{}".format(i)))
df = pd.concat(b, axis=1)

# Show data sample
df[df.columns[:2]].plot()

# Get features
y = anomalous.ts_measures(df, freq=30)

# Show features in principal components
idx_name = y.index.name or "index"
y = y.reset_index().set_index([idx_name, 'variable'])
anomalous.biplot_features(y)

# Detect anomalies
anomalous.anomaly(y)
```

GPL Licence

Credits to [Rob J Hyndman](https://github.com/robjhyndman)