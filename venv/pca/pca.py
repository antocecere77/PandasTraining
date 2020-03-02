# La Principal Component Analysis (PCA) è la tecnica di riduzione della dimensionalità non supervisionata più diffusa.

# In sistesi la PCA consiste nell'identificare le direzioni di maggiore varianza nei dati, che vengono chiamate
# componenti pricipali (dall'inglese principal components). La prima componente principale è la direzione di maggior
# varianza, le altre sono le direzioni di maggior varianza ortogonali alle precedenti. L'ortogonalità ci assicura che
# ogni componente sia indipendente dalle altre e quindi contenga informazioni differente.

# La PCA, come le altre tecniche di riduzione della dimensionalità, può essere utilizzata principlamente per due scopi,
# visualizzare i dati in 2 o 3 dimensioni oppure velocizzare la fase di addestramento, in questo notebook li vedremo
# entrambi.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
import time
import seaborn as sns

# PCA per creare visualizzazioni
# Il dataset che utilizzeremo è il Wine Dataset, un dataset di vini contenente 13 proprietà che permettono di suddivedere i vini in 3 differenti categorie.
#Un dataset con 13 proprietà ha 13 dimensioni, ma noi possiamo visualizzare i dati in 2, massimo 3, dimensioni. Applichiamo la PCA al Wine Dataset per eseguire una riduzione dimensionale da 13 a 2 dimensioni.

# Cominciamo importando il dataset dalla repository.

cols = ["label","alcol","acido malico","cenere","alcalinità della cenere","magnesio",
        "fenoli totali","flavonoidi","fenoli non-flavonoidi","proantocianidine",
        "intensità del colore","tonalità", "OD280/OD315 dei vini diluiti","prolina"]


wines = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
                 names=cols)

print(wines.head())
print(wines.var(axis=0))

# Settiamo seaborn per renderle più eleganti le visualizzazioni che andremo a creare.
sns.set()

# Creiamo gli array numpy, dato che il nostro scopo è solo quello di visualizzare i dati e non creare un modello
# possiamo anche non dividerli in set di addestramento e test.
X = wines.drop("label", axis=1).values
Y = wines["label"].values

# Prima di applicare la PCA dobbiamo essere SICURISSIMI che i dati siano su una scala comune, quindi eseguiamo la
# standardizzazione.
sc = StandardScaler()
X = sc.fit_transform(X)

# Adesso possiamo eseguire la Principal Component Analysis utilizzando la classe PCA di sklearn specificando il numero
# di componenti desiderate, cioè 2, all'interno del costruttore.

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pc = pca.fit_transform(X)

# Utilizziamo matplotlib per visualizzare le due componenti su di uno scatter plot.

plt.xlabel("Prima componente principale")
plt.ylabel("Seconda componente principale")
plt.scatter(X_pc[:,0], X_pc[:,1], c=Y, edgecolors='black')
plt.show()

# Nonostante la riduzione da 13 a 2 proprietà la suddivisione delle classi all'interno del dataset è ben visibile.

