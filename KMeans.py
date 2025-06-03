
''' k-means :
    Un algorithme non spervisé pour classifier les données en K clusters.
    le k-means necessite de choisir au préalable le nombre de clusters.
    cette implémentation est a but de ségmentation.
'''
"""
    Algo :
    1 - choix de K clusters aléatoires
    2 - affectation des pixels : chaque pixel est affecté au cluster le plus proche (selon un critère/métrique tell que la distance euclidienne)
    3 - mise à jour des centroïdes : moyenne des points de chaque cluster
    4 - répéter jusqu'à convergence
    
    Les données sont : une matrice de 3 caractéristiques (RGB), donc de taille (image_size x 3)
"""

import numpy as np


# Fonction 1
def k_means(X, K, max_iter=100, tol=1e-4):
    """
    Args:
        X (float): pixels de l'image
        K (int): nombre de clusters
        max_iter (int): nombre d'itération max
        tol (float)

    Returns:
        labels : affectations des donées aux clusters
        C : centroides finaux
    """
    
    # initialisation des centroides (aleatoire)
    ids = np.random.choice(X.shape[0], K, replace=False) #choix d'indexes 
    C = X[ids] #centroides
    for _ in range(max_iter):
        # Calcul de distances 
        C_O = C.copy()
        dist = np.linalg.norm(X[:, None] - C[None, :], axis=2) #celcul vectorisé, pour ne pas faire une boucle
        labels = np.argmin(dist, axis=1)
        # MAJ des controides
        for k in range(K):
            C[k] = X[labels == k].mean(axis=0)
        if np.linalg.norm(C - C_O) < tol :
            break
    return labels, C

# Fonction 2 : plutot avec des boucles 

def kMeans1(X, K, max_iter=100, tol=1e-4):
    ids = np.random.choice(X.shape[0], K, replace=False) #choix d'indexes 
    C = X[ids] #centroides
    for _ in range(max_iter):
        labels = np.zeros((X.shape[0]))
        C_O = C.copy()
        for i in range(X.shape[0]):
            dist = np.zeros((K))
            for k in range(K):
                dist[k] = np.linalg.norm(X[i] - C[k]) #au début j'avais oublié de faire la norma haha
            labels[i] = np.argmin(dist)
        for k in range(K):
            C[k] = X[labels == k].mean(axis=0)
        if np.linalg.norm(C - C_O) < tol :
            break
    return labels, C 
        
    
# Version objet :)
class KMeans :
    def __init__(self, K=3, max_iter=100, tol=1e-4):
        self.K=K
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None # à initialiser plutard dans fit
        self.labels_= None
    # méthode pour entrainer le modèle
    def fit(self, X):
        
        self.X = X
        id = np.random.choice(X.shape[0], self.K, replace=False)
        self.centroids = X[id]
        for _ in range(self.max_iter):
            C = self.centroids.copy()
            dist = np.linalg.norm(X[:, None] - C[None, :], axis=2) # (NxK)
            self.labels_ = np.argmin(dist, axis=1)
            
            for k in range(self.K):
                self.centroids[k] = X[self.labels_ == k].mean(axis=0)
            if np.linalg.norm(self.centroids - C) < self.tol :
                break
        return self.centroids
    
    # méthode pour prédire une nouvelle classe
    def predict(self, X):
        dist = np.linalg.norm(X[:, None] - self.centroids[None, :], axis=2)
        labels = np.argmin(dist, axis=1)
        return labels
        