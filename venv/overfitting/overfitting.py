# Overfitting e regolarizzazione
# L'overfitting è un problema tipico del machine learning che si manifesta quando un modello si lega troppo ai dati
# di addestramento e fallisce nel generalizzare su dati nuovi.
#
# L'overffiting è caratterizzato da:
#
# Alta variaza: le previsioni per modelli addestrati con diverse parti del dataset saranno molto diverse tra loro.
# Basso bias: l'errore per le predizioni sul set di addestramento è mediamente molto basso.
#
# Per riconoscere un caso di overfitting confronto l'errore sul set di addestramento con quello sul set di test,
# se quest'ultimo è molto più grande mi trovo davanti ad un caso di overfitting

import pandas as pd
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

boston = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data", sep='\s+',
                     names=["CRIM", "ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PRATIO","B","LSTAT","MEDV"])
print(boston.head())

X = boston.drop('MEDV',axis=1).values
Y = boston['MEDV'].values

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=0)

poly_feats = PolynomialFeatures(degree=2)
X_train_poly = poly_feats.fit_transform(X_train)
X_test_poly = poly_feats.transform(X_test)

print(X_train_poly.shape)

ss = StandardScaler()
X_train_poly = ss.fit_transform(X_train_poly)
X_test_poly = ss.transform(X_test_poly)

ll = LinearRegression()
ll.fit(X_train_poly, Y_train)

Y_pred_train = ll.predict(X_train_poly)

mse_train = mean_squared_error(Y_train, Y_pred_train)
r2_train = r2_score(Y_train, Y_pred_train)

print("Train set: MSE=" + str(mse_train) + " R2=" + str(r2_train))

Y_pred_test = ll.predict(X_test_poly)

mse_test = mean_squared_error(Y_test, Y_pred_test)
r2_test = r2_score(Y_test, Y_pred_test)

print("Test set: MSE=" + str(mse_test) + " R2=" + str(r2_test))