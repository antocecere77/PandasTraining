# La Kernel PCA è una variante della Principal Component Analysis che si serve delle funzioni kernel e il Kernel Trick,
# che abbiamo già trattato qui, per passare da uno spazio dimensionale non lineare a uno minore lineare.

# Osserivamo il funzionamento della Kernel PCA su di un dataset contenente una relazione non lineare tra le sue 2
# proprietà.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score, log_loss

# Costruiamo il nostro dataset ad hoc utilizzando la funzione make_circle di scikit-learn.

X, y = make_circles(n_samples=1000, noise=0.1, factor=0.2, random_state=1)
plt.scatter(X[:,0],X[:,1],c=y)
plt.show()

# Non importa che classificatore utilizziamo, se questo è lineare non riuscirà mai a dividere bene le due classi di
# questo dataset.
# Non ci credi ? Proviamo ad eseguire una regressione logistica.
# Realizziamo due funzioni per creare il modello e visualizzare il decision boundary, in modo da poterle riutilizzare.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

lr = LogisticRegression()
lr.fit(X_train, y_train)

print(accuracy_score(y_train, lr.predict(X_train)))
print(accuracy_score(y_test, lr.predict(X_test)))

# Ho 0.5, in pratica ho sbagliato il 50% dei casi

from scripts.viz import plot_boundary

plot_boundary(lr, X, y)

# Applicare la Kernel PCA
# Osserviamo adesso come la situazione cambia drasticamente quando utilizziamo la Kernel PCA.
# Quando il dataset assume questa forma circolare il kernel da utilizzare è quello gaussiano (o radial basis function
# - RBF), in generale il kernel gaussiano è quello da utilizzare sempre quando non sappiamo come muoverci.
# Applichiamo la Kernel PCA utilizzando la classe KernelPCA di sklearn.

from sklearn.decomposition import KernelPCA

kpca = KernelPCA(kernel="rbf", gamma=5)
kpc = kpca.fit_transform(X)

plt.scatter(kpc[:,0],kpc[:,1], c=y)
plt.xlabel("Kernel Principal Component 1")
plt.ylabel("Kernel Principal Component 2")
plt.show()

# La Kernel PCA ha estratto due componenti principali, la prima contiene le informazioni necessarie per eseguire la
# classificazione con un semplice classificatore lineare.
# Proviamo a visualizzare solo la prima componente.

plt.xlabel("Kernel Principal Component 1")
plt.scatter(kpc[:,0],np.zeros((1000,1)), c=y)
plt.show()

# Adesso il problema di classificazione è estremamente banale, proviamo con un semplice regressore lineare.
fpc = kpc[:,0]
fpc = fpc.reshape(-1,1)
print(fpc.shape)

X_train, X_test, y_train, y_test = train_test_split(fpc, y, test_size=0.2, random_state=0)

lr = LogisticRegression()
lr.fit(X_train, y_train)

print(accuracy_score(y_train, lr.predict(X_train)))
print(accuracy_score(y_test, lr.predict(X_test)))