# La fase di ottimizzazione degli iperparametri, conosciuta in gergo tecnico come hyperparameters tuning, è la parte più
# ostica nel processo di creazione di un modello predittivo.
# In questo notebook vedremo due tecniche che ci permettono di semplificare questo lavoro.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# Carichiamo l'Iris Dataset all'interno di un DataFrame.

iris = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                   names=["sepal length","sepal width","petal length","petal width","class"])
print(iris.head())

X = iris.drop("class",axis=1).values
Y = iris["class"].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

svc = SVC()

param_grid = {"kernel": ["linear","rbf","sigmoid","poly"],
             "C": [1, 10, 100, 1000],
             "gamma": [0.1,1,"auto"]}

gs = GridSearchCV(svc, param_grid, cv=10) #in cv specifichiamo il numero di folds per la cross-validation

start_time = time.time()
gs.fit(X_train, Y_train)
print(time.time() - start_time)

print("Iperparametri del modello migliore: %s" % gs.best_params_)
print("Accuracy=%.4f" % gs.best_score_)

# Iperparametri del modello migliore: {'C': 1, 'gamma': 0.1, 'kernel': 'linear'}
# Accuracy=0.9810
# Possiamo anche ottenere direttamente il modello migliore già addestrato tramite l'attributo best_estimator.
# Recuperiamo il modello migliore trovato durante la grid search e calcoliamo l'accuracy sul test set.

svc = gs.best_estimator_
print(svc.score(X_test,Y_test))

# La seconda tecnica che vediamo è la Random Search, che ricerca i valori ottimali degli iperparametri cercandoli a caso
# in una distribuzione di valori da noi definita.
# Evidenze sperimentali hanno dimostrato che questo approccio porta a risultati migliori e più velocemente rispetto alla
# Grid Search.
# Possiamo implementare la Random Search utilizzando la classe RandomizedSearchCV di scikit-learn.
# Questa classe effettua la random search su una distribuzione di parametri che definiamo tramite un dizionario e valida
# i modelli ottenuti tramite k-fold cross-validation.

from sklearn.model_selection import RandomizedSearchCV

svc = SVC()

param_dist = {"kernel": ["linear","rbf","sigmoid","poly"],
             "C": [1, 10, 100, 1000],
             "gamma": [0.1,1,"auto"]}

rs = RandomizedSearchCV(svc, param_dist, cv=10)
start_time = time.time()
rs.fit(X_train, Y_train)
print(time.time() - start_time)

print("Iperparametri del modello migliore: %s" % gs.best_params_)
print("Accuracy=%.4f" % gs.best_score_)