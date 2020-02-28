# Le reti neurali artificiali sono un modello del machine learning estremamente potente quanto complesso, che funzionano
# replicando il funzionamento del cervello.Il percettrone multistrato è un'architettura di rete neurale cositutia da una
# rete di neuroni (chiamati anche nodi) distribuiti su più layers:

# Un layer di input: in cui il numero di neuroni corrisponde al numero di proprietà del nostro dataset.
# Un layer di output: in cui il numero di neuroni corrisponde al numero di classi.
# Uno o più hidden layers: livelli intermedi che utilizzano l'output del layer precendente per apprendere nuove
# proprietà.
# Una rete neurale con un solo hidden layer è anche definita vanilla neural network.
# Una rete neurale con due o più hidden layers è anche definita deep neural network (rete neurale profonda)
# Ogni neurone di un hidden layer corrisponde ad un percettrone e l'attivazione di ognuno di essi è data da una funzione
# di attivazione non lineare.

# Per essere addestrata correttamente una rete neurale richiede un numero elevato di esempi per l'addestramento, per
# questo utilizzeremo il MNIST dataset, un dataset di immagini di cifre scritte a mano contenente 60.000 esempi per
# l'addestramento e 10.000 per il test.

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

from mnist import load_mnist

X_train, X_test, Y_train, Y_test = load_mnist(path="MNIST")
print("Numero totale di proprietà: "+str(X_train.shape[1]))
print("Esempi di training: "+str(X_train.shape[0]))
print("Esempi di test: "+str(X_test.shape[0]))

# Le immagini hanno una dimensione di 28x28 pixels, quindi un esempio ha 784 proprietà.
# Eseguiamo la normalizzazione di train set e test set.

mms = MinMaxScaler()
X_train = mms.fit_transform(X_train)
X_test = mms.transform(X_test)

# Proviamo ad utilizzare una regressione logistica per classificare gli esempi del MNIST.
# (Potrebbe richiedere diversi minuti)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=.1)
lr.fit(X_train,Y_train)

y_pred_train = lr.predict(X_train)
y_prob_train = lr.predict_proba(X_train)

y_pred = lr.predict(X_test)
y_prob = lr.predict_proba(X_test)

accuracy_train = accuracy_score(Y_train, y_pred_train)
accuracy_test = accuracy_score(Y_test, y_pred)

loss_train = log_loss(Y_train, y_prob_train)
loss_test = log_loss(Y_test, y_prob)

print("ACCURACY: TRAIN=%.4f TEST=%.4f" % (accuracy_train,accuracy_test))
print("LOG LOSS: TRAIN=%.4f TEST=%.4f" % (loss_train,loss_test))

# Vanilla Neural Network
# Adesso creiamo un percettrone multistrato con un unico hidden layer contenente 100 neuroni.
# Utilizziamo la classe MLPClassifier di scikit-learn, questa classe ha un numero di parametri da far girare la testa, cosa normale per una rete neurale, noi utilizziamo soltanto il parametro hidden_layer_sizes, all'interno del quale possiamo specificare il numero di hidden layers e di nodi per ognuno di essi.
# Settiamo il parametro verbose a True per visualizzare l'addestramento in maniera dinamica. (Potrebbe richiedere diversi minuti)

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(100,), verbose=True)
mlp.fit(X_train, Y_train)

y_pred_train = mlp.predict(X_train)
y_prob_train = mlp.predict_proba(X_train)

y_pred = mlp.predict(X_test)
y_prob = mlp.predict_proba(X_test)

accuracy_train = accuracy_score(Y_train, y_pred_train)
accuracy_test = accuracy_score(Y_test, y_pred)

loss_train = log_loss(Y_train, y_prob_train)
loss_test = log_loss(Y_test, y_prob)

print("ACCURACY: TRAIN=%.4f TEST=%.4f" % (accuracy_train, accuracy_test))
print("LOG LOSS: TRAIN=%.4f TEST=%.4f" % (loss_train, loss_test))

for i in range(0,len(X_test)):
    if(Y_test[i]!=y_pred[i]):
        print("Numero %d classificato come %d" % (Y_test[i], y_pred[i]))
        plt.imshow(X_test[i].reshape([28,28]), cmap="gray")
        plt.show()