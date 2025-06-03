
"""
   Algorithme du clustering agglomératif : clustering hierarchique ascendant. L'idée est de trouver mieux qu le k-means qui
   est sensible a l'initialisation et donne différents clusters a chaque execution de l'algo, si l'on veux augmenter le nombre de clusters ou le diminuer, on peut sans 
   repasser par un autre apprentissage.
   il ne suppose rien sur la forme des clusters (donc peuvent etre de forme quelconque)
   
   idée/algo :
   - On commence avec chaque point comme étant son propre cluster
   - à chaque étape, on fusionne les deux clusters les plus proches (fin on regroupe en un cluster)
   - on répète jusqu'a n'avoir qu'un seul cluster
   résultat : dendogramme = structure arborescente
"""

"""
Algo :
    1 - calcul de la matrice de distances entre tous les points
    2 - trouver les 2 clusters les plus proches
    3 - les fusionner
    4 - recalculer les distances
    5 - répéter jusqu'à obtenir un seul cluster
    
    - chaque cluster est une liste d'indicss
    - on a une liste de distances entre les clusters (t pas seulement les points)
    - une boucle qui :
        * trouve les deux clusters les plus proches (en dépends du linkage)
        * les fusionne
        * met à jour la liste + les distance
    Le linkage utilisé de base sera "single linkage", et qu'on étendra plutard vers les autres.
"""
import numpy as np

def single_linkage_distance(C1, C2, X):
    return min(np.linalg.norm(X[i] - X[j]) for i in C1 for j in C2) 

def complete_linkage_distance(C1, C2, X):
    return max(np.linalg.norm(X[i] - X[j]) for i in C1 for j in C2)


# pour single linkage
def agglomerative_clustering(X, K):
    # init
    clusters = [[i] for i in range(len(X))]
    while len(clusters)>K :
        min_dist = float('inf') #infinity
        merge_pair = (0,1)
        # la on va itérer sur les clusters et calculer les distances
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                dist = single_linkage_distance(clusters[i], clusters[j], X)
                if dist < min_dist :
                    min_dist = dist
                    merge_pair = (i, j)
        i, j = merge_pair
        new_cluster = clusters[i] + clusters[j]
        clusters.pop(i)
        clusters.pop(j)
        clusters.append(new_cluster)
    
    labels = np.zeros(len(X), dtype=int)
    for idx, cluster in enumerate(clusters):
        for i in cluster : # car clusters 
            labels[i] = idx
    
    return labels, clusters 


# Encapsulation 

class AgglomerativeClustering :
    def __init__(self, K, linkage):
        self.K = K
        self.linkage = linkage
        self.clusters_ = None
        self.labels_ = None # le "_" signifie que c'est un attribut généré par le modèle aprés entrainement (convension en python askip) = résultat d'apprentissage et pas des paramèttres d'entrée
    def _get_linkage(self):
        if self.linkage == 'single':
            return lambda c1, c2, X: min(np.linalg.norm(X[i] - X[j]) for i in c1 for j in c2) # une autre facon de faire :)
        if self.linkage == 'complete':
            return lambda c1, c2, X: max(np.linalg.norm(X[i] - X[j]) for i in c1 for j in c2) 
        if self.linkage == 'average':
            return lambda c1, c2, X: np.mean([np.linalg.norm(X[i] - X[j]) for i in c1 for j in c2]) # ni le min, ni le max, mais plutot la moyenne
        else:
            raise ValueError(f"linkage '{self.linkage}' n'est pas supporté")
    def fit(self, X):
        self.X = X #besoin pour le test aprés
        linage_func = self._get_linkage()
        clusters = [[i] for i in range(len(X))]
        while len(clusters)>self.K:
            min_dist = float('inf')
            merge_pair = (0, 1)
            for i in range(len(clusters)):
                for j in range(i+1, len(clusters)): # parcour de chaque paire de clusters
                    dist = linage_func(clusters[i], clusters[j], X)
                    if dist < min_dist:
                        min_dist = dist
                        merge_pair = (i, j)
            i, j = merge_pair
            c1 = clusters[i]
            c2 = clusters[j]
            for index in sorted([i, j], reverse=True):
                clusters.pop(index)
            clusters.append(c1 + c2)
        labels = np.zeros(len(X), dtype=int)
        for idx, cluster in enumerate(clusters):
            for i in cluster:
                labels[i] = idx
        self.clusters_ = clusters
        self.labels_ = labels
        
        return self #ici on obtient un modèle alors (donc un objet aprés apprentissage)
    
    def predict(self, X_test):
        linkage_func = self._get_linkage() 
        labels = np.zeros(len(X_test), dtype=int)
        
        for idx, x in enumerate(X_test):
            # ici on va mettre l'emement a la fin de X_train : pour pouvoir utiliser la fonction de linkage
            data = np.vstack([self.X, x])
            indice = len(data)-1
            
            min_dist = float('inf')
            best_cluster = -1
            
            for i, cluster in enumerate(self.clusters_):
                dist = linkage_func([indice], cluster, data)
                if dist<min_dist:
                    min_dist = dist
                    best_cluster = i 
            labels[idx] = best_cluster
        return labels # that was absolutely smart!!! :))
    
    
            
        