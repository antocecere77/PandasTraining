# Eseguirò una classificazione utilizzando Full Batch Gradient Descend, Stochastic Gradient Descend e Mini Batch
# Gradient Descend al fine di confrontare le oscillazioni della funzione di costo per ognuno di essi.

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import log_loss

# Utilizziamo la funzione make_classification di scikit-learn per creare un dataset ad-hoc in cui poter eseguire la
# nostra classificazione

X, Y = make_classification(n_samples=1250, n_features=4, n_informative=2, random_state=0)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=0)

# Lo Stochastic Gradient Descend (SGD) è una versione del Gradient Descend che utilizza un solo esempio di addestramento
# per volta durante l'ottimizzazione.
#
# Possiamo utilizzare la classe SGDClassifier di scikit-learn, di default questa utilizza lo SGD per addestrare una SVM,
# ma possiamo modificare la tipologia di modello semplicemente cambiando la funzione di costo:
#
# hinge: l'SGD addestrerà una SVM
# log: l'SGD addestrerà una regressione logistica
# perceptron: l'SGD addestrerà un percettrone
# Per evitare cicli bisogna mischiare il train set ad ogni epoca, per farlo basta impostare il parametro shuffle a True
# (questo valore è settato di default, quindi è possibile anche ometterlo)
#
# NOTA BENE Lo Stochastic Gradient Descend è un processo stocastico, anche utilizzando gli stessi iperparametri un
# modello fornirà risultati lievemente differenti se riaddestrato.

from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle


sgd = SGDClassifier(loss="log", verbose=False, shuffle=True) #verbose ci mostrerà informazioni sull'avanzamento dell'addestramento
sgd.fit(X_train,Y_train)
print("LOSS: %.4f" % (log_loss(Y_test, sgd.predict_proba(X_test))))

def minibatchGD(train_set, test_set, n_batches=1, epochs=10):
    X_train, Y_train = train_set
    X_test, Y_test = test_set

    batch_size = X_train.shape[0]/n_batches

    sgd = SGDClassifier(loss="log")
    sgd_loss = []

    for epoch in range(epochs):
        X_shuffled, Y_shuffled = shuffle(X_train, Y_train)
        for batch in range(n_batches):
            batch_start = int(batch*batch_size)
            batch_end = int((batch+1) * batch_size)

            x_batch = X_shuffled[batch_start:batch_end, :]
            y_batch = Y_shuffled[batch_start:batch_end]

            sgd.partial_fit(x_batch, y_batch, classes=np.unique(Y_train))
            loss = log_loss(Y_test, sgd.predict_proba(X_test), labels=np.unique(Y_train))

            sgd_loss.append(loss)

            print("Loss all'epoca %d = %.4f" %(epoch+1, loss))
    return (sgd, sgd_loss)

full_gd, full_gd_loss = minibatchGD((X_train, Y_train), (X_test, Y_test), n_batches=1, epochs=200)

sgd, sgd_loss = minibatchGD((X_train, Y_train), (X_test, Y_test), n_batches=X_train.shape[0], epochs=5)

mini_gd, mini_gd__loss = minibatchGD((X_train, Y_train), (X_test, Y_test), n_batches=10, epochs=50)

plt.rcParams['figure.figsize']=(14,10)
plt.plot(sgd_loss, label="Stochastic")
plt.show()

plt.plot(mini_gd__loss, label="Mini Batch")
plt.show()

plt.plot(full_gd_loss, label="Full Batch")
plt.show()

plt.plot(sgd_loss, label="Stochastic")
plt.plot(mini_gd__loss, label="Mini Batch")
plt.plot(full_gd_loss, label="Full Batch")
plt.xlim(xmin=0, xmax=200)
plt.legend()
plt.show()
