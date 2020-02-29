# Clustering Gerarchico
# Il Clustering Gerarchico è un metodo di clustering che affronta il problema creando una gerarchia di cluster.
# Esistono due tipologie di Clustering Gerarchico:

# Agglomerativo e' un approccio bottom-up, in cui si parte creando un cluster per ogni esempio e gradualmente si
# uniscono due a due, fino ad ottenere un unico cluster.
# Divisivo è un approccio top-down, in cui si parte con un cluster che contiene tutti gli esempi che viene man mano
# suddiviso ricorsivamente in sotto-clusters.

# In questo notebook utilizzeremo il Clustering Gerarchico Agglomerativo per raggruppare in cluster un dataset generato
# da noi.

# Importiamo matplotlib e seaborn, impostiamo la dimensione di default dei grafici e settiamo lo stile di seaborn.

import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["figure.figsize"] = (14, 10)
sns.set()

# Creiamo il nostro dataset, composto da 100 esempi che possono essere suddivisi in 3 clusters.

from sklearn.datasets.samples_generator import make_blobs

X, y_true = make_blobs(n_samples=100, centers=3,
                       cluster_std=.5, random_state=0)

plt.scatter(X[:, 0], X[:, 1], s=50);
plt.show()

# Creare un dendrogramma con scipy
# Possiamo utilizzare scipy per eseguire il clustering gerarchico agglomerativo di un dataset, in questo modo otterremo un dendrogramma che conterrà tutti i possibili cluster.

# Per farlo utilizziamo le seguenti funzioni:

# linkage: per eseguire il clustering gerarchico
# dendrogram: per costruire il dendrogramma

from scipy.cluster.hierarchy import linkage, dendrogram

plt.figure(figsize=(18,14))
dendogram = dendrogram(linkage(X, method="ward"))
plt.ylabel("Distanza")
plt.title("Dendrogramma")
plt.xticks(fontsize=8)
plt.show()

# Possiamo eseguire il Clustering Gerarchico Agglomerativo anche utilizziando scikit-learn, ma in questo caso dobbiamo
# specificare il numero di cluster come parametro del modello, quindi a priori.

# Dalla precedente analisi del dendrogramma abbiamo determinato che il numero di clusters ottimale è 3, quindi possiamo
# passare a costruire il modello, per farlo utilizziamo la classe AgglomerativeClustering di scikit-learn a cui dobbiamo
# passare il numero di clusters da cercare all'interno del parametro n_clusters.

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=3)
y = hc.fit_predict(X)

# Stampiamo il risultato del clustering su di uno scatter plot.

plt.scatter(X[:, 0], X[:, 1], c=y, s=200, cmap='viridis', edgecolors="black");
plt.show()