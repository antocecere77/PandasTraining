import pandas as pd
import numpy as np

shirts = pd.read_csv("data/shirts.csv", index_col=0)
print(shirts.head())

X = shirts.values
print(X[:10])

# Sostituisco la label con un numero
# dizionario che ordina le misure
size_mapping = {"S":0,"M":1,"L":2,"XL":3}

shirts["taglia"] = shirts["taglia"].map(size_mapping)
print(shirts.head())

fmap = np.vectorize(lambda t:size_mapping[t])
# Applico l'fmap alla prima colonna
X[:,0] = fmap(X[:,0])

print(X[:5])

# One hot encoding (ogni tipo di colore ha una sua proprietà con true o false)
shirts = pd.get_dummies(shirts,columns=["colore"])
print(shirts.head())

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

X = shirts.values
transf = ColumnTransformer([('ohe', OneHotEncoder(), [1])], remainder="passthrough")

X = transf.fit_transform(X)

print(X[:10])