import pandas as pd

iris = pd.read_csv("data/iris.csv")

print(iris.head(10))
print(iris.tail(10))

iris = pd.read_csv("data/iris_noheader.csv", header=None, names=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])
print(iris.head(10))
print(iris.tail(10))

print(iris.columns)
print(iris.info())

Y = iris['species']
print(Y.head())

print(type(Y))

x = iris[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
print(x.head())

#axis righe 1, colonne 0
x = iris.drop("species", axis=1)
print(x.head())

iris_sampled = iris.copy()
iris_sampled = iris.sample(frac=1)
print(iris_sampled.head())

print("iloc")
print(iris_sampled.iloc[3]) #seleziona la riga alla posizione 10

print("loc")
print(iris_sampled.loc[112])

print(iris_sampled.loc[32, "species"])

print(iris_sampled.iloc[:10,1])

print(iris.shape)

print(iris.describe())

print("Massima lunghezza del petalo")
print (iris['petal_length'].max())
print("Minima lunghezza del petalo")
print (iris['petal_length'].min())
print("Lunghezza del petalo media")
print (iris['petal_length'].mean())
print("Mediana della lunghezza del petalo")
print (iris['petal_length'].median())
print("Varianza della lunghezza del petalo")
print (iris['petal_length'].var())
print("Deviazione standard della lunghezza del petalo")
print (iris['petal_length'].std())

print(iris['species'].unique())

#creiamo una maschera per selezionare solo le osservazioni i cui petali sono più lunghi della media
long_petal_mask = iris['petal_length'] > iris['petal_length'].mean()
print(long_petal_mask)

iris_long_petals = iris[long_petal_mask]
print(iris_long_petals.head())

iris_copy = iris.copy()

iris_copy[iris_copy["species"]=="setosa"]="undefined"
print(iris_copy["species"].unique())

#normalizzazione delle features numeriche
X = iris.drop("species", axis=1)
X_norm = (X - X.min())/(X.max()-X.min())
print(X_norm.head())

print(iris.sort_values("petal_length").head())

grouped_species = iris.groupby("species")
print(grouped_species.mean())

import numpy as np

print(iris.apply(np.count_nonzero, axis=1).head())

X = iris.drop("species", axis=1)
X = X.applymap(lambda val: int(round(val, 0)))
print(X.head())

iris_nan = iris.copy()
samples = np.random.randint(iris.shape[0], size=(10))
iris_nan.loc[samples, "petal_length"] = None

print(iris_nan["petal_length"].isnull().sum())

mean_petal_length = iris_nan["petal_length"].mean()

iris_nan["petal_length"].fillna(mean_petal_length, inplace=True)
print(iris_nan["petal_length"].isnull().sum())

import matplotlib.pyplot as plt
iris.plot(x="sepal_length", y="sepal_width", kind="scatter")
plt.show()