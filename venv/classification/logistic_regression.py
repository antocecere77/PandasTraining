import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

breast_cancer = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
names=["id","diagnosis","radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean",
       "concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se",
       "perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se",
       "fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst",
       "compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"])

print(breast_cancer.info())

print(breast_cancer["diagnosis"].unique())

X = breast_cancer[["radius_se", "concave points_worst"]].values
Y = breast_cancer["diagnosis"].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# Trasformo le stringhe M e B in numeri 0,1
le = LabelEncoder()
Y_train = le.fit_transform(Y_train)
Y_test = le.transform(Y_test)

print(Y_train[:5])

# Standardizzo il dataset
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, Y_train)

from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss

Y_pred = lr.predict(X_test)
Y_pred_proba = lr.predict_proba(X_test)

print("ACCURACY: " + str(accuracy_score(Y_test, Y_pred)))
print("LOG LOSS: " + str(log_loss(Y_test, Y_pred_proba)))

from viz import show_bounds

show_bounds(lr, X_train, Y_train, labels=["Benigno", "Maligno"])
show_bounds(lr, X_test, Y_test, labels=["Benigno", "Maligno"])

# Provo ad utilizzare tutte le variabili

X = breast_cancer.drop(["diagnosis", "id"], axis=1).values
Y = breast_cancer["diagnosis"].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# Trasformo le stringhe M e B in numeri 0,1
le = LabelEncoder()
Y_train = le.fit_transform(Y_train)
Y_test = le.transform(Y_test)

# Standardizzo il dataset
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

lr = LogisticRegression()
lr.fit(X_train, Y_train)

Y_pred = lr.predict(X_test)
Y_pred_proba = lr.predict_proba(X_test)

print("ACCURACY: " + str(accuracy_score(Y_test, Y_pred)))
print("LOG LOSS: " + str(log_loss(Y_test, Y_pred_proba)))

lr = LogisticRegression()


