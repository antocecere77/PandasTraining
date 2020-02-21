import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score

from sklearn.datasets import load_digits

digits = load_digits()

X = digits.data
Y = digits.target

print(X.shape)

print(np.unique(Y))

for i in range(0, 10):
    pic_matrix = X[Y==i][0].reshape([8,8])
    plt.imshow(pic_matrix, cmap="gray")
    plt.show()

X_train, X_test,Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()
X_train = mms.fit_transform(X_train)
X_test = mms.transform(X_test)

lr = LogisticRegression()
lr.fit(X_train, Y_train)

Y_pred = lr.predict(X_test)
Y_pred_proba = lr.predict_proba(X_test)

print("ACCURACY: " + str(accuracy_score(Y_test, Y_pred)))
print("LOG LOSS: " + str(log_loss(Y_test, Y_pred_proba)))

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, Y_pred)
print(cm)

import seaborn as sns

plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, cmap="Blues_r", linewidths=.5, square=True)
plt.ylabel("Classe corretta")
plt.xlabel("Classe predetta")
plt.show()

from sklearn.multiclass import OneVsRestClassifier

ovr = OneVsRestClassifier(LogisticRegression()) # Utilizziamo la regressione logistica come classificatore
ovr.fit(X_train, Y_train)

y_pred_proba = ovr.predict_proba(X_test)
y_pred = ovr.predict(X_test)

print("ACCURACY: "+str(accuracy_score(Y_test, y_pred)))
print("LOG LOSS: "+str(log_loss(Y_test, y_pred_proba)))