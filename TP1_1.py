#Imports
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import cluster
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn import datasets
from sklearn.metrics import davies_bouldin_score
from scipy.io import arff
#%%
#1. Dataset
from scipy.io import arff
path = './artificial/'
databrut = arff.loadarff(open(path+"R15.arff",'r'))
data = [[x[0],x[1]] for x in databrut[0]]

#Affichage 2D
f0 = [f[0] for f in data]
f1 = [f[1] for f in data]
plt.scatter(f0,f1,s=8)
plt.title("Données initiales")
plt.show()
#%%
#2. Clustering k-Means & k-Medoids
#2.1 Pour démarrer
print("Appel kMeans pour une valeur fixée de k")
tps1 = time.time()
k = 3
model = cluster.KMeans(n_clusters=k, init='k-means++')
model.fit(data)
tps2=time.time()
labels=model.labels_
iteration=model.n_iter_

plt.scatter(f0, f1, c=labels, s=8)
plt.title("Données après clustering Kmeans")
plt.show()
print("nb clusters=",k," nb_iter=",iteration, "runtime=",round((tps2-tps1)*1000,2),"ms")
#%%
#2.2 Intérêts de la méthode k-Means
# Automatically determine the best number of clusters
#Testing different evaluation metrics
k_list = []
score_sil = []
score_dav = []
runtime_list=[]
best=0
best_k=0
for k in range(2,25):
    tps1 = time.time()
    model = cluster.KMeans(n_clusters=k, init='k-means++')
    model.fit(data)
    tps2=time.time()
    labels=model.labels_
    iteration=model.n_iter_
    
    plt.scatter(f0, f1, c=labels, s=8)
    plt.title("Données après clustering Kmeans")
    plt.show()
    k_list.append(k)
    runtime_list.append(round((tps2-tps1)*10,2))
    #Silouhette index: the higher the index the better the clustering
    score_sil.append(metrics.silhouette_score(data, labels, metric='euclidean'))
    
    #David Bouldin index: the lower the index the better the clustering
    #score_dav.append(davies_bouldin_score(data, labels))
    
    if (score_sil[k-2] > best) :
        best = score_sil[k-2]
        best_k=k
    

plt.plot(k_list,score_sil)
#plt.plot(k_list,score_dav)
plt.plot(k_list,runtime_list)
print(best,best_k)
#%%
# 2.3 : Limits k-Means
# Dataset

path = './artificial/'
databrut = arff.loadarff(open(path+"spiral.arff",'r'))
data = [[x[0],x[1]] for x in databrut[0]]

#Affichage 2D
f0 = [f[0] for f in data]
f1 = [f[1] for f in data]
plt.scatter(f0,f1,s=8)
plt.title("Données initiales")
plt.show()

#2.2 Intérêts de la méthode k-Means
# Automatically determine the best number of clusters
#Testing different evaluation metrics
k_list = []
score_sil = []
score_dav = []
runtime_list=[]
best=0
best_k=0
for k in range(2,35):
    tps1 = time.time()
    model = cluster.KMeans(n_clusters=k, init='k-means++')
    model.fit(data)
    tps2=time.time()
    labels=model.labels_
    iteration=model.n_iter_
    
    plt.scatter(f0, f1, c=labels, s=8)
    plt.title("Données après clustering Kmeans")
    plt.show()
    k_list.append(k)
    runtime_list.append(round((tps2-tps1)*10,2))
    #Silouhette index: the higher the index the better the clustering
    score_sil.append(metrics.silhouette_score(data, labels, metric='euclidean'))
    
    #David Bouldin index: the lower the index the better the clustering
    #score_dav.append(davies_bouldin_score(data, labels))
    
    if (score_sil[k-2] > best) :
        best = score_sil[k-2]
        best_k=k
    

plt.plot(k_list,score_sil)
#plt.plot(k_list,score_dav)
plt.plot(k_list,runtime_list)
print(best,best_k)

# For this dataset, the greater the cluster number the greater the score returned by silouhette method, which is not true because 
# the optimal number of clusters is 2


#%%
# 2.4 :
import kmedoids