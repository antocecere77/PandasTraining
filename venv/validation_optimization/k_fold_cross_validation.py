# La k-fold cross-validation è un metodo di validazione del modello che non influisce sulla dimensione del train set.
# In questo notebook creeremo un modello di classificazione per l'iris dataset e utilizzeremo una 10-folds
# cross-validation per la validazione.

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

iris = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                   names=["sepal length","sepal width","petal length","petal width","class"])
print(iris.head())

X = iris.drop("class",axis=1).values
Y = iris["class"].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

# Per eseguire una cross-validation con scikit-learn abbiamo 2 opzioni
#
# K-folds Cross Validation method
# Utilizziamo la classe KFold di sklearn per creare 10 folds e addestrare un modello per ognuna di esse.

from sklearn.model_selection import KFold

lr = LogisticRegression()
kfold = KFold(n_splits=10, random_state=1)
scores = []

for k, (train, test) in enumerate(kfold.split(X_train)):
    lr.fit(X_train[train], Y_train[train])
    score = lr.score(X_train[test],Y_train[test])
    scores.append(score)
    print("Fold %d: Accuracy=%.2f" %(k, score))

accuracy = np.array(scores).mean()
print("\nValidation Accuracy = %.2f" % accuracy)

from sklearn.model_selection import StratifiedKFold

lr = LogisticRegression()

kfold = StratifiedKFold(n_splits=10, random_state=1)
scores = []

for k, (train, test) in enumerate(kfold.split(X_train, Y_train)):
    lr.fit(X_train[train], Y_train[train])
    score = lr.score(X_train[test], Y_train[test])
    scores.append(score)
    print("Fold %d: Accuracy=%.2f" % (k, score))

print("\nValidation Accuracy = %.2f" % (np.array(scores).mean()))

# La seconda opzione che scikit-learn offre è più semplice e più efficace.
# E' possibile utilizzare la cross_validation come funzione di scoring passandogli il modello, questa funzione utilizza
# internamente la StratifiedKFold quindi le classi saranno bilanciate.

from sklearn.model_selection import cross_val_score

lr = LogisticRegression()
scores = cross_val_score(lr, X_train, Y_train, cv=10)

for fold, score in enumerate(scores):
    print("Fold %d score=%.4f" % (fold + 1, score))

print("\nValidation Accuracy = %.2f" % scores.mean())