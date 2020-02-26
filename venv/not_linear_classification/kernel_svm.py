# Le macchine a vettori di supporto sono un modello linerare, quindi falliscono nel trovare relazioni non lineari
# all'interno dei dati.
# Questa limitazione può essere superata utilizzando una funzione kernel.
# In questo notebook testeremo diverse funzioni kernel per classificare un toy-dataset contenente due features legate
# da una relazione non linere.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from viz import plot_bounds

# Per creare il dateset possiamo utilizzare il metodo make-circle di scikit-learn, questo ritornerà features e target
# già sotto forma di array numpy.

from sklearn.datasets import make_circles

X, Y = make_circles(noise=0.2, factor=0.5, random_state=1)
print(X.shape)
plt.scatter(X[:,0],X[:,1],c=Y)
plt.show()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

from sklearn.svm import SVC

svc = SVC(kernel="linear", probability=True)
svc.fit(X_train, Y_train)

print("ACCURACY: Train=%.4f Test=%.4f" % (svc.score(X_train, Y_train), svc.score(X_test,Y_test)))
plot_bounds((X_train, X_test),(Y_train, Y_test),svc)

# Il kernel gaussiano è il kernel generico più utilizzato ed è quello da utilizzare se non si sa come muoversi.

from sklearn.svm import SVC

svc = SVC(kernel="rbf",probability=True) #equivale alla classe LinearSVC
svc.fit(X_train, Y_train)
print("ACCURACY: Train=%.4f Test=%.4f" % (svc.score(X_train, Y_train), svc.score(X_test,Y_test)))
plot_bounds((X_train, X_test),(Y_train, Y_test),svc)

# Kernel alternativo che vale la pena provare se RBF non fornisce risultati soddisfacenti (non è questo il caso)

from sklearn.svm import SVC

svc = SVC(kernel="sigmoid",probability=True) #equivale alla classe LinearSVC
svc.fit(X_train, Y_train)
print("ACCURACY: Train=%.4f Test=%.4f" % (svc.score(X_train, Y_train), svc.score(X_test,Y_test)))
plot_bounds((X_train, X_test),(Y_train, Y_test),svc)

# Altro kernel alternativo che vale la pena provare se RBF non fornisce risultati soddisfacenti (continua a non essere
# questo il caso)

from sklearn.svm import SVC

svc = SVC(kernel="poly",probability=True) #equivale alla classe LinearSVC
svc.fit(X_train, Y_train)
print("ACCURACY: Train=%.4f Test=%.4f" % (svc.score(X_train, Y_train), svc.score(X_test, Y_test)))
plot_bounds((X_train, X_test),(Y_train, Y_test),svc)