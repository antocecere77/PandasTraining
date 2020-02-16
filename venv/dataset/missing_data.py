import pandas as pd
import numpy as np

iris_nan = pd.read_csv("data/iris_nan.csv")

Y = iris_nan["class"].values
print(Y)

X = iris_nan.drop("class", axis=1).values
print(iris_nan)

# Soluzione non consigliata
iris_drop = iris_nan.dropna(axis=1)
print(iris_drop)

# Soluzione imputazione dati mancanti con media
replace_with = iris_nan.mean()
iris_imp = iris_nan.fillna(replace_with)
print(iris_imp)

replace_with = iris_nan.dropna().mode().iloc[0]
iris_imp = iris_nan.fillna(replace_with)
print(iris_imp)


import numpy as np
from sklearn.impute import SimpleImputer

imp = SimpleImputer(missing_values = np.nan, strategy = 'mean')
X_imp = imp.fit_transform(X)
print(X_imp)