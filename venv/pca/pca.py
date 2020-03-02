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

# Quante componenti principali selezionare ?
# Nell'esempio, volendo costruire una visualizzazione, la scelta era forzata a 2 o 3, ma in altri casi vorremmo poter
# selezionare il numero di componenti principali che ci permettono di ridurre la complessità del modello mantenendo la
# maggior quantità possibile di informazione (cioè di varianza).

# Per avere un'idea possiamo visualizzare graficamente la percentuale di varianza contenuta da ogni componente insieme
# alla varianza comulativa di tutte le possibili componenti, o di una parte di esse.

# Per prima cosa dobbiamo prima creare tutte le componenti principali, possiamo farlo impostando il parametro n_
# components a None.


from sklearn.decomposition import PCA

pca = PCA(n_components=None)
pc = pca.fit(X)
print(pca.explained_variance_ratio_)

# Adesso in pca sono presenti le 13 componenti principali, explained_variance_ratio_ contiene la percentuale della
# varianza totale che ogni singola componente contiene.

# Utilizziamo un grafico a barre in combinazione con uno step chart per visualizzare la varianza per ogni singola
# componente e la varianza cumulativa. Per eseguire una somma cumulativa possiamo usare la funzione cumsum di Numpy.

plt.bar(range(1, 14), pca.explained_variance_ratio_, align='center') # varianza per singola componente
plt.step(range(1, 14), np.cumsum(pca.explained_variance_ratio_), where='mid') #varianza cumulativa
plt.ylabel('Varianza')
plt.xlabel('Componenti principali')
plt.show()