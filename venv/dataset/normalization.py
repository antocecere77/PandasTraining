import pandas as pd
import numpy as np

wines = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
                    names=['classe','alcol','flavonoidi'],
                    usecols=[0,1,7])
print(wines.head())

Y = wines['classe'].values
X = wines.drop('classe', axis=1).values
print(X)
print(Y)

print(wines.describe())

# Applico la normalizzazione
wines_norm = wines.copy()
features = ["alcol", "flavonoidi"]
to_norm = wines[features]

wines_norm[features] = (to_norm - to_norm.min())/(to_norm.max() - to_norm.min())
print(wines_norm.head())

# Normalizzazione di un dataframe
from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()
X_norm = X.copy()
X_norm = mms.fit_transform(X_norm)
print(X_norm[:5])

# Standardidazione

wines_std = wines.copy()
to_std = wines_std[features]

wines_std[features] = (to_std - to_std.mean())/to_std.std()
print(wines_std[:5])

from sklearn.preprocessing import StandardScaler

X_std = X.copy()
ss = StandardScaler()
X_std = ss.fit_transform(X_std)
print(X_std[:5])