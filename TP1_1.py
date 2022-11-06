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
import kmedoids
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
import scipy.cluster.hierarchy as shc
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
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
tps2=time.time()s1 = time.time()
model = cluster.AgglomerativeClustering(linkage = 'single',
                                        n_clusters = k )
model = model.fit(data)
tps2 = time.time()
labels = model.labels_
kres = model.n_clusters_
leaves = model.n_leaves_
# Affichage clustering
plt.scatter(f0,f1,c = labels,s = 8 )
plt.title ("Resultat du clustering ")
plt.show()
print (" nb clusters = " ,k , " , nb feuilles = " , leaves ," runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )

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
k_list = []s1 = time.time()
model = cluster.AgglomerativeClustering(linkage = 'single',
                                        n_clusters = k )
model = model.fit(data)
tps2 = time.time()
labels = model.labels_
kres = model.n_clusters_
leaves = model.n_leaves_
# Affichage clustering
plt.scatter(f0,f1,c = labels,s = 8 )
plt.title ("Resultat du clustering ")
plt.show()
print (" nb clusters = " ,k , " , nb feuilles = " , leaves ," runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )

score_sil = []
score_dav = []
runtime_list=[]
best=0from sklearn import metrics

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


#2.2 Intérêts de la méthode k-Means
# Automatically determine the best number of clusters
#Testing different evaluation metrics
k_list = []
score_sil = []
score_dav = []s1 = time.time()
model = cluster.AgglomerativeClustering(linkage = 'single',
                                        n_clusters = k )
model = model.fit(data)
tps2 = time.time()
labels = model.labels_
kres = model.n_clusters_
leaves = model.n_leaves_
# Affichage clustering
plt.scatter(f0,f1,c = labels,s = 8 )
plt.title ("Resultat du clustering ")
plt.show()
print (" nb clusters = " ,k , " , nb feuilles = " , leaves ," runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )

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
    runtime_list.append(round((tps2-tps1),2))
    #Silouhette index: the higher the index the better the clustering
    score_sil.append(metrics.silhouette_score(data, labels, metric='euclidean'))
    
    #David Bouldin index: the lower the index the better the clustering
    #score_dav.append(davies_bouldin_score(data, labels))
    
    if (score_sil[k-2] > best) :
        best = score_sil[k-2]
        best_k=k
    

plt.plot(k_list,score_sil,label ='Score Silhouette')
#plt.plot(k_list,score_dav)
plt.plot(k_list,runtime_list,label ='Runtime')
plt.legend()
print(best,best_k)

# For this dataset, the greater the cluster number the greater the score returned by silouhette method, which is not true because 
# the optimal number of clusters is 2


#%%
# 2.4 :

tps1 = time.time()
k = 3
distmatrix = euclidean_distances( data )
fp = kmedoids.fasterpam( distmatrix , k )
tps2 = time.time()
iter_kmed = fp.n_iter
labels_kmed = fp.labels
print ( " Loss with FasterPAM : " , fp.loss )
plt . scatter ( f0 , f1 , c = labels_kmed , s = 8 )
plt . title ( " Donnees apres clustering KMedoids " )
plt . show ()
print ( " nb clusters = " ,k , " , nb iter = " , iter_kmed , " ,runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )

#%%
#Testing different evaluation metrics
k_list = []
score_sil = []
score_dav = []
runtime_list=[]
best=0
best_k=0
for k in range(2,20):
    tps1 = time.time()
    
    distmatrix = euclidean_distances( data )
    fp = kmedoids.fasterpam( distmatrix , k )
    tps2 = time.time()
    iter_kmed = fp.n_iter
    labels_kmed = fp.labels
    print ( " Loss with FasterPAM : " , fp.loss )
    plt . scatter ( f0 , f1 , c = labels_kmed , s = 8 )
    plt . title ( " Donnees apres clustes1 = time.time()
model = cluster.AgglomerativeClustering(linkage = 'single',
                                        n_clusters = k )
model = model.fit(data)
tps2 = time.time()
labels = model.labels_
kres = model.n_clusters_
leaves = model.n_leaves_
# Affichage clustering
plt.scatter(f0,f1,c = labels,s = 8 )
plt.title ("Resultat du clustering ")
plt.show()
print (" nb clusters = " ,k , " , nb feuilles = " , leaves ," runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )
ring KMedoids " )
    plt . show ()
    print ( " nb clusters = " ,k , " , nb iter = " , iter_kmed , " ,runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )
        
    
    k_list.append(k)
    runtime_list.append(round((tps2-tps1),2))
    
    
    #Silouhette index: the higher the index the better the clustering
    score_sil.append(metrics.silhouette_score(data, labels_kmed, metric='euclidean'))
    
    #David Bouldin index: the lower the index the better the clustering
    #score_dav.append(davies_bouldin_score(data, labels))
    if (score_sil[k-2] > best) :
        best = score_sil[k-2]
        best_k=k

plt.plot(k_list,score_sil,label ='Score Silhouette')
#plt.plot(k_list,score_dav)
plt.plot(k_list,runtime_list,label ='Runtime')
print(best,best_k)
plt.legend()

#%%

#Rand score k_means and kmedoids , k=15
# K-means
print("Appel kMeans pour une valeur fixée de k")
tps1 = time.time()
k = 15
model = cluster.KMeans(n_clusters=k, init='k-means++')
model.fit(data)
tps2=time.time()
labels_k_means=model.labels_
iteration=model.n_iter_

plt.scatter(f0, f1, c=labels_k_means, s=8)
plt.title("Données après clustering Kmeans")
plt.show()
print("nb clusters=",k," nb_iter=",iteration, "runtime=",round((tps2-tps1)*1000,2),"ms")


# kmedoids
tps1 = time.time()
k=15
distmatrix = euclidean_distances( data )
fp = kmedoids.fasterpam( distmatrix , k )
tps2 = time.time()s1 = time.time()
model = cluster.AgglomerativeClustering(linkage = 'single',
                                        n_clusters = k )
model = model.fit(data)
tps2 = time.time()
labels = model.labels_
kres = model.n_clusters_
leaves = model.n_leaves_
# Affichage clustering
plt.scatter(f0,f1,c = labels,s = 8 )
plt.title ("Resultat du clustering ")
plt.show()
print (" nb clusters = " ,k , " , nb feuilles = " , leaves ," runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )

iter_kmed = fp.n_iter
labels_kmed = fp.labels
print ( " Loss with FasterPAM : " , fp.loss )
plt . scatter ( f0 , f1 , c = labels_kmed , s = 8 )
plt . title ( " Donnees apres clustering KMedoids " )
plt . show ()
print ( " nb clusters = " ,k , " , nb iter = " , iter_kmed , " ,runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )


# Rand_score application
print("rand_score: ",metrics.rand_score(labels_k_means, labels_kmed))
#%%
#3.CLustering Agglomératif
#3.1
# Donnees dans datanp
print ("Dendrogramme * single * donnees initiales " )
linked_mat = shc.linkage ( data , 'single')
plt.figure(figsize = (12,12))
shc.dendrogram(linked_mat,
               orientation = 'top' ,
               distance_sort = 'descending' ,
               show_leaf_counts = False)
plt.show()
#%%
#set distance_threshold ( 0 ensures we compute the full tree )
#Testing different distances
linkage = ['single','average','complete', 'ward']
single_t = []
average_t = []
complete_t = []
ward_t = []

for l in linkage:
    print("Linkage ",l)
    for i in range(1, 200,4):
        tps1 = time.time()
        model = cluster.AgglomerativeClustering(distance_threshold = i*0.1,
                                                linkage = l,
                                                n_clusters = None )
        model = model.fit(data)
        tps2 = time.time()
        labels = model.labels_
        k = model.n_clusters_
        leaves =model.n_leaves_
        # Affichage clustering
        plt.scatter(f0,f1,c = labels,s = 8 )
        plt.title ("Resultat du clustering ")
        plt.show()
        print (" nb clusters = " ,k , " , nb feuilles = " , leaves ," runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms ", "distance = ",i*0.1 )
        if (l == 'single') :
            single_t.append(round (( tps2 - tps1 ) * 1000 , 2 ))
        if (l == 'average') :
            average_t.append(round (( tps2 - tps1 ) * 1000 , 2 ))
        if (l == 'complete') :
            complete_t.append(round (( tps2 - tps1 ) * 1000 , 2 ))
        if (l == 'ward') :
            ward_t.append(round (( tps2 - tps1 ) * 1000 , 2 ))
        if (k == 1) :
            break
        
print ("single : ", single_t)
print ("average : ", average_t)
print ("complete : ", complete_t)
print ("ward : ", ward_t)
#%%
# set the number of clusters
k = 4
tps1 = time.time()
model = cluster.AgglomerativeClustering(linkage = 'single',
                                        n_clusters = k )
model = model.fit(data)
tps2 = time.time()
labels = model.labels_
kres = model.n_clusters_
leaves = model.n_leaves_
# Affichage clustering
plt.scatter(f0,f1,c = labels,s = 8 )
plt.title ("Resultat du clustering ")
plt.show()
print (" nb clusters = " ,k , " , nb feuilles = " , leaves ," runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )

# same distance between clusters : aglomératif > kmeans
# nuage de données : aglomeratif < kmeans

#%%
#4 Clustering DBSCAN et HDBSCAN
#4.1

clustering = DBSCAN(eps=3, min_samples=1).fit(data)
labels = clustering.labels_
kres = clustering.
plt.scatter(f0,f1,c = labels,s = 8 )
plt.title ("Resultat du clustering ")
plt.show()

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print("Estimated number of clusters: %d" % n_clusters_)

# Distances k plus proches voisins
# Donnees dans X
k = 5
neigh = NearestNeighbors( n_neighbors = k )
neigh.fit(data)
distances , indices = neigh.kneighbors(data)
# retirer le point " origine "
newDistances = np.asarray([np.average(distances[i][1:] ) for i in range (0 , distances.shape[0])])
trie = np.sort(newDistances)
#plt.title("Plus proches voisins (5)")
#plt.plot(trie) ;
#plt.show()























