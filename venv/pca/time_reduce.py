# PCA per ridurre il tempo di addestramento
# Adesso vediamo come utilizzare la PCA per velocizzare la fase di addestramento comprimendo il numero di proprietà,
# cercando allo stesso tempo di mantenere buona parte della varianza.
# A questo scopo utilizzeremo il MNIST dataset, il dataset di cifre scritte a mano, nella sua versione integrale con
# 70.000 esempi e 784 proprietà, una per ogni pixel dell'immagine.

# Se non lo hai già fatto puoi scaricare il dataset da questo sito, devi scaricare i seguenti 4 files:

# train-images-idx3-ubyte.gz
# train-labels-idx1-ubyte.gz
# t10k-images-idx3-ubyte.gz
# t10k-labels-idx1-ubyte.gz
# poi decomprimili e inseriscili all'interno di una cartella "MNIST", senza rinominarli.
# I files con le proprietà contengono immagini in formato binario, quindi ho scritto una funzione che ti permette di
# ottenere direttamente gli array di train e test da questi files (la puoi trovare all'interno del file mnist.py)

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression

from scripts.mnist import load_mnist
import matplotlib.pyplot as plt

X_train, X_test, Y_train, Y_test = load_mnist(path="MNIST")
print("Numero totale di proprietà: "+str(X_train.shape[1]))
print("Esempi di training: "+str(X_train.shape[0]))
print("Esempi di test: "+str(X_test.shape[0]))

from matplotlib.pyplot import imshow

print(Y_test[0])
imgplot = imshow(X_test[0].reshape(28, 28), cmap='gray')
plt.show()

# Adesso dobbiamo portare i dati sulla stessa scala, che ripeto è FONDAMENTALE per il funzionamento del PCA, trattandosi
# di immagini dobbiamo eseguire la normalizzazione.

mms = MinMaxScaler()
X_train = mms.fit_transform(X_train)
X_test = mms.transform(X_test)

from time import time
from sklearn.decomposition import PCA

pca = PCA(0.90)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print(X_test_pca.shape)

lr = LogisticRegression(solver='lbfgs')
start_time = time()
lr.fit(X_train_pca, Y_train)
end_time = time()

print(end_time - start_time)

print(accuracy_score(Y_train, lr.predict(X_train_pca)))
print(accuracy_score(Y_test, lr.predict(X_test_pca)))
print(log_loss(Y_train, lr.predict_proba(X_train_pca)))
print(log_loss(Y_test, lr.predict_proba(X_test_pca)))
