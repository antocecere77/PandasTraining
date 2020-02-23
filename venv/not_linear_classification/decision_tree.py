# In questo tutorial utilizzò alberi decisionali (dall'inglese Decision Tree) e foreste casuali (dall'inglese Random
# Forest) per creare un modello di machine learning in grado di dirci se una determinata persona avrebbe avuto
# possibilità di salvarsi dal disastro del Titanic.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score

titanic = pd.read_csv("http://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv")
titanic.info()

print(titanic.head())

titanic = titanic.drop("Name", axis=1)
titanic = pd.get_dummies(titanic)

print(titanic.head())

X = titanic.drop("Survived", axis=1).values
Y = titanic["Survived"].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
print(X_train.shape)

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion="gini", max_depth=6)
tree.fit(X_train, Y_train)

y_pred_train = tree.predict(X_train)
y_pred = tree.predict(X_test)

accuracy_train = accuracy_score(Y_train, y_pred_train)
accuracy_test = accuracy_score(Y_test, y_pred)

print("ACCURACY: TRAIN=%.4f TEST=%.4f" % (accuracy_train,accuracy_test))

from sklearn.tree import export_graphviz

dotfile = open("tree.dot", 'w')
export_graphviz(tree, out_file = dotfile, feature_names = titanic.columns.drop("Survived"))
dotfile.close()

# Il  file tree.dot può essere visualizzato su questo sito o installando graphviz
