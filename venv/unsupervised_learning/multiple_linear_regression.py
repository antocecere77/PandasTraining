# Regressione lineare multipla utilizzando il Boston Housing Dataset.
# Il Boston Housing Dataset contiene 506 esempi di abitazioni nella zona di Boston con le seguenti 14 features:
# CRIM Tasso di criminalità per capita
# ZN Percentuale di terreni residenziali suddivisi in zone per lotti superiori a 25.000 sq.ft.
# INDUS Percentuale di ettari di attività non al dettaglio per città.
# CHAS Variabile dummy che indica la prossimità al fiume Charles.
# NOX Concentrazione di ossido d'azoto (parti per 10 milioni).
# RM Numero medio di stanze per abitazione
# AGE Percentuale di abitazione occupate costruite dopo il 1940
# DIS Media pesata delle distanze da 5 centri lavorativi di Boston.
# RAD Indice di accessibilità ad autostrade
# TAX Aliquota dell'imposta sulla proprietà a valore pieno in 10.000 USD.
# PRATIO Rapporto studente-insegnante per città.
# BLACK 1000(Bk - 0.63)^2 dove Bk è la percentuale di abitanti di colore per città
# LSTAT Percentuale della popolazione povera
# MEDV Mediana del valore di abitazioni occupate in 1.000 USD.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

boston = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data", sep='\s+',
                     names=["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PRATIO","B","LSTAT","MEDV"])
print(boston.head())

print(boston.info())

# Vedo le correlazioni
# -1 correlazione inversa: all'aumentare di una colonna ne seguirà il diminuire dell'altra
# 0 nulla o scarsa
# 1 correlazione diretta
print(boston.corr())

# Per vederle meglio creo una heatmap
import seaborn as sns
hm = sns.heatmap(boston.corr(),
                 yticklabels=boston.columns, #labels per i valori sull'asse Y
                 xticklabels=boston.columns) #labels per i valori sull'asse X
plt.show()

cols = ["RM", "LSTAT", "PRATIO", "TAX", "INDUS", "MEDV"]
hm = sns.heatmap(boston[cols].corr(),
                 yticklabels=boston[cols].columns,
                 xticklabels=boston[cols].columns,
                 annot=True,                          #Questo ci mostra i valori degli indici
                 annot_kws={'size':12})               #Impostiamo la dimensione dell'annotazione a 12 per farla entrare dentro il quadrato

plt.show()

sns.pairplot(boston[cols])
plt.show()

X = boston[["RM", "LSTAT"]].values
Y = boston["MEDV"].values

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=0)

ll = LinearRegression()
ll.fit(X_train, Y_train)
Y_pred = ll.predict(X_test)

print("MSE: "+str(mean_squared_error(Y_test, Y_pred)))
print("R2 score: "+str(r2_score(Y_test, Y_pred)))

# Il test va decisamente meglio, utilizzo adesso tutte le proprietà

X = boston.drop("MEDV", axis=1).values
Y = boston["MEDV"].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
X_train_std = ss.fit_transform(X_train)
X_test_std = ss.transform(X_test)

ll = LinearRegression()
ll.fit(X_train_std, Y_train)
Y_pred = ll.predict(X_test_std)

print("MSE: "+str(mean_squared_error(Y_test, Y_pred)))
print("R2 score: "+str(r2_score(Y_test, Y_pred)))

print(list(zip(boston.columns, ll.coef_)))