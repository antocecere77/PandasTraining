import pandas as pd
import numpy as np

boston = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
                     sep="\s+", usecols=[5,13], names=["RM", "MEDV"])

print(boston.head())

X = boston.drop("MEDV", axis=1).values
Y = boston["MEDV"].values

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

from sklearn.linear_model import LinearRegression

ll = LinearRegression()
ll.fit(X_train, Y_train)
Y_pred = ll.predict(X_test)

from sklearn.metrics import mean_squared_error

print(mean_squared_error(Y_test, Y_pred))

from sklearn.metrics import r2_score

# <0.3 inutle
# >0.3 <0.5 scarso
# >0.5 <0.7 discreto
# >0.7 <0.9 buono
# >0.9 <1 ottimo
# 1 perfetto predice sempre il risultato, è probabile che ci sia un errore
print("Score: " + str(r2_score(Y_test, Y_pred)))

import matplotlib.pyplot as plt

print("Peso di RM: " + str(ll.coef_[0]))
print("Bias: " + str(ll.intercept_))

plt.scatter(X_train, Y_train, c="green", edgecolors="white", label="Train set")
plt.scatter(X_test, Y_test, c="blue", edgecolors="white", label="Test set")

plt.xlabel("Numero medio di stanze [RM]")
plt.ylabel("Valore in $1000 [MEDV]")

plt.legend(loc="upper left")
plt.plot(X_test, Y_pred, color="red", linewidth=3)

plt.show()