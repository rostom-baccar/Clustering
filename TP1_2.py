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
import hdbscan
from sklearn.neighbors import NearestNeighbors
from scipy.io import arff
import pandas as pd

#%%
def load_arff_data (name ) :
    path = './Bureau/TP-Clustering/artificial/'
    databrut = arff.loadarff(open(path+name+".arff",'r'))
    data = [[x[0],x[1]] for x in databrut[0]]

    #Affichage 2D
    f0 = [f[0] for f in data]
    f1 = [f[1] for f in data]
    plt.scatter(f0,f1,s=8)
    plt.title("Données initiales "+name)
    plt.show()
    d={}
    d["data"] = data
    d["f0"] = f0
    d["f1"] = f1
    
    return d
    
#%%
loaded_r15 = load_arff_data("R15")
data_r15 = loaded_r15["data"]
f0_r15 = loaded_r15["f0"]
f1_r15 = loaded_r15["f1"]

loaded_d31 = load_arff_data("D31")
data_d31 = loaded_d31["data"]
f0_d31 = loaded_d31["f0"]
f1_d31 = loaded_d31["f1"]

loaded_simplex = load_arff_data("simplex")
data_simplex = loaded_simplex["data"]
f0_simplex = loaded_simplex["f0"]
f1_simplex = loaded_simplex["f1"]

loaded_spiral = load_arff_data("spiral")
data_spiral = loaded_spiral["data"]
f0_spiral = loaded_spiral["f0"]
f1_spiral = loaded_spiral["f1"]

loaded_donut1 = load_arff_data("donut1")
data_donut1 = loaded_donut1["data"]
f0_donut1 = loaded_donut1["f0"]
f1_donut1 = loaded_donut1["f1"]


#%%
#2. Clustering k-Means & k-Medoids
#2.1 Pour démarrer

# Function that performs the kmeans algorithm
def kmeans_iteration( loaded_data , k ):
    #print("Appel kMeans pour une valeur fixée de k")
    tps1 = time.time()
    
    model = cluster.KMeans(n_clusters=k, init='k-means++')
    model.fit(loaded_data["data"])
    tps2=time.time()
    labels = model.labels_
    iteration=model.n_iter_
    
    plt.scatter(loaded_data["f0"], loaded_data["f1"], c=labels, s=8)
    plt.title("Clustering KMeans - k = "+str(k))
    plt.show()
    #print("nb clusters=",k," nb_iter=",iteration, "runtime=",round((tps2-tps1)*1000,2),"ms")
    
    d={}
    d["labels"]=labels
    d["iteration"]= iteration
    d["tps"]= round((tps2-tps1)*1000,2)
    
    return d


#%%
#2.2 Intérêts de la méthode k-Means
# Automatically determine the best number of clusters

#Function that given a method, tests different numbers of clusters and plots the corresponding score
def kmeans_evaluation_graph(method,loaded_data,max_k):
    k_list = []
    score_sil = []
    score_dav = []
    runtime_list=[]
    best=0
    best_k=0
    
    for i in range(2,max_k):
        
        if (method == "kmeans") :
            km_return = kmeans_iteration(loaded_data,i)
        elif (method == "kmedoids"):
            km_return = kmedoids_iteration(loaded_data,i)
        else : 
            print("Method not recognized ! ")
            break
        
        k_list.append(i)
        runtime_list.append(km_return["tps"]*0.01)
        
        #Silouhette index: the higher the index the better the clustering
        score_sil.append(metrics.silhouette_score(loaded_data["data"], km_return["labels"], metric='euclidean'))
        
        #Davies Bouldin index: the lower the index the better the clustering
        score_dav.append(davies_bouldin_score(loaded_data["data"], km_return["labels"]))
        
        if (score_sil[i-2] > best) :
            best = score_sil[i-2]
            best_k=i
        
    plt.plot(k_list,score_sil,label ='Score Silhouette')
    plt.plot(k_list,score_dav, label ='Score Davies-Bouldin')
    plt.plot(k_list,runtime_list,label ='Runtime (10**-1 s)')
    plt.legend()
    print(best,best_k)

#25

#kmeans_evaluation_graph("kmeans",loaded_r15,25)


#%%

# 2.3 : Limits k-Means
# Dataset spiral

#kmeans_evaluation_graph("kmeans",loaded_spiral,35)

# For this dataset, the greater the cluster number the greater the score returned by silouhette method, which is not true because 
# the optimal number of clusters is 2


#%%
# 2.4 :Méthode k-medoids

# Function that performs the kmedoids algorithm
def kmedoids_iteration ( loaded_data , k ):
    tps1 = time.time()

    distmatrix = euclidean_distances( loaded_data["data"] )
    fp = kmedoids.fasterpam( distmatrix , k )
    tps2 = time.time()
    iter_kmed = fp.n_iter
    labels_kmed = fp.labels
    #print ( " Loss with FasterPAM : " , fp.loss )
    plt . scatter ( loaded_data["f0"] , loaded_data["f1"] , c = labels_kmed , s = 8 )
    plt . title ( "Clustering KMedoids - k = "+str(k) )
    plt . show ()
    #print ( " nb clusters = " ,k , " , nb iter = " , iter_kmed , " ,runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )
    
    d={}
    d["labels"]=labels_kmed
    d["iteration"]= iter_kmed
    d["tps"]= round((tps2-tps1)*1000,2)
    
    return d



#%%
#Testing different evaluation metricson kmedoids
#kmeans_evaluation_graph("kmedoids",loaded_r15,20)


#%%

#Rand score k_means and kmedoids , k=15

# K-means
#kmeans_return = kmeans_iteration(loaded_r15, 3)

# kmedoids
#kmedoids_return = kmedoids_iteration(loaded_r15, 3)

# Rand_score application
print("rand_score: ",metrics.rand_score(kmeans_return["labels"], kmedoids_return["labels"]))


#%%

#3.CLustering Agglomératif
#3.1
# Donnees dans datanp
print ("Dendrogramme * single * donnees initiales " )
linked_mat = shc.linkage ( data_r15 , 'single')
plt.figure(figsize = (12,12))
shc.dendrogram(linked_mat,
               orientation = 'top' ,
               distance_sort = 'descending' ,
               show_leaf_counts = False)
plt.show()
#%%

# Function that performs the agglomerative algorithm
def agglomerative_iteration (loaded_data , linkage, distance=None, k=None):
    
    tps1 = time.time()
    model = cluster.AgglomerativeClustering(distance_threshold = distance,
                                            linkage = linkage,
                                            n_clusters = k )
    model = model.fit(loaded_data["data"])
    tps2 = time.time()
    labels = model.labels_
    k = model.n_clusters_
    leaves =model.n_leaves_
    # Affichage clustering
    plt.scatter(loaded_data["f0"],loaded_data["f1"],c = labels,s = 8 )
    plt.title ("Clustering Agglomératif - Mode "+linkage+ " distance = "+str(distance))
    plt.show()
    print (" nb clusters = " ,k , " , linkage = " , linkage ," , nb feuilles = " , leaves ," runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms ", "distance = ",distance )
    
    d={}
    d["tps"]=round (( tps2 - tps1 )*1000,2 )
    d["k"]=k
    d["labels"]=labels
    d["leaves"]=leaves
    
    return d
    
#ret = agglomerative_iteration(loaded_spiral,"single",distance = 0.46)

#%%
#set distance_threshold ( 0 ensures we compute the full tree )
#Testing different distances

#Function that given a linkage and a max_distance, tests different distances and plots the corresponding score
def agglomerative_evaluation_graph_distance (loaded_data, linkage, max_distance): 
    d_list = []
    score_sil = []
    score_dav = []
    runtime_list=[]
    best=0
    best_k=0
    
    for i in range(1,max_distance*10):
        
        agg_return = agglomerative_iteration(loaded_data,linkage,distance = i*0.1)
        
        d_list.append(i*0.1)
        runtime_list.append(agg_return["tps"]*0.01)
        
        #Silouhette index: the higher the index the better the clustering
        score_sil.append(metrics.silhouette_score(loaded_data["data"], agg_return["labels"], metric='euclidean'))
        
        #Davies Bouldin index: the lower the index the better the clustering
        score_dav.append(davies_bouldin_score(loaded_data["data"], agg_return["labels"]))
        
        if (score_sil[i-2] > best) :
            best = score_sil[i-2]
            best_k=i
        
    plt.plot(d_list,score_sil,label ='Score Silhouette')
    plt.plot(d_list,score_dav, label ='Score Davies-Bouldin')
    plt.plot(d_list,runtime_list,label ='Runtime (10**-1 s)')
    plt.title("Evaluation selon la distance")
    plt.legend()
    print(best,best_k)

#agglomerative_evaluation_graph_distance(loaded_r15,'single',2)

#%%

#Function that given a linkage and a max_k, tests different different number of clusters and plots the corresponding score
def agglomerative_evaluation_graph_k (loaded_data, linkage, max_k): 
    k_list = []
    score_sil = []
    score_dav = []
    runtime_list=[]
    best=0
    best_k=0
    
    for i in range(2,max_k):
        
        agg_return = agglomerative_iteration(loaded_data,linkage,k = i)
        
        k_list.append(i)
        runtime_list.append(agg_return["tps"]*0.01)
        
        #Silouhette index: the higher the index the better the clustering
        score_sil.append(metrics.silhouette_score(loaded_data["data"], agg_return["labels"], metric='euclidean'))
        
        #Davies Bouldin index: the lower the index the better the clustering
        score_dav.append(davies_bouldin_score(loaded_data["data"], agg_return["labels"]))
        
        if (score_sil[i-2] > best) :
            best = score_sil[i-2]
            best_k=i
        
    plt.plot(k_list,score_sil,label ='Score Silhouette')
    plt.plot(k_list,score_dav, label ='Score Davies-Bouldin')
    plt.plot(k_list,runtime_list,label ='Runtime (10**-1 s)')
    plt.title("Evaluation selon le nombre de cluster")
    plt.legend()
    print(best,best_k)

#agglomerative_evaluation_graph_k(loaded_r15,'single',20)

#%%
# set the number of clusters
#ret = agglomerative_iteration(loaded_spiral,"single",distance = None, k=2)

# same distance between clusters : aglomératif > kmeans
# nuage de données : aglomeratif < kmeans

#%%
#4 Clustering DBSCAN et HDBSCAN
#4.1

# Function that performs the dbscan algorithm
def dbscan_iteration (loaded_data , eps , min_samples):
    
    tps1 = time.time()
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(loaded_data["data"])
    tps2 = time.time()
    labels = clustering.labels_
    # kres = clustering.n_clusters_
    plt.scatter(loaded_data["f0"],loaded_data["f1"],c = labels,s = 8 )
    plt.title ("Clustering DBSCAN - eps = "+str(eps)+" min_samples = "+str(min_samples))
    plt.show()
    
    k = len(set(labels)) - (1 if -1 in labels else 0)
    print("For eps = %f and min_Estimated = %d, number of clusters: %d" % (eps, min_samples,k))
    d={}
    d["tps"]=round (( tps2 - tps1 )*1000, 2 )
    d["k"]=k
    d["labels"]=labels
    
    return d
    
#%%
# Testing DBSCAN method with different values for min-sample and eps
for min_samples in range(2,10):
    for eps in range(1,20,2):
        dbscan_iteration(loaded_r15, eps*0.1, min_samples)
        

#%%

# Distances k plus proches voisins
def neighbors_eps(loaded_data , k):
    neigh = NearestNeighbors( n_neighbors = k )
    neigh.fit(loaded_data["data"])
    distances , indices = neigh.kneighbors(loaded_data["data"])
    # retirer le point " origine "
    newDistances = np.asarray([np.average(distances[i][1:] ) for i in range (0 , distances.shape[0])])
    trie = np.sort(newDistances)
    plt.title("Plus proches voisins (5)")
    plt.plot(trie) ;
    plt.show()


#neighbors_eps(loaded_spiral,5)
#%%
#d = dbscan_iteration(loaded_spiral, 0.30, 5)

#%%
#4 Clustering DBSCAN et HDBSCAN
#4.1

# Function that performs the dbscan algorithm
def hdbscan_iteration (loaded_data , min_samples):
    
    tps1 = time.time()
    clustering = hdbscan.HDBSCAN(min_samples=min_samples).fit(loaded_data["data"])
    tps2 = time.time()
    labels = clustering.labels_
    # kres = clustering.n_clusters_
    plt.scatter(loaded_data["f0"],loaded_data["f1"],c = labels,s = 8 )
    plt.title ("Clustering HDBSCAN - min_samples = "+str(min_samples))
    plt.show()
    
    k = len(set(labels)) - (1 if -1 in labels else 0)
    print("For eps = %f and min_Estimated = %d, number of clusters: %d" % (eps, min_samples,k))
    d={}
    d["tps"]=round (( tps2 - tps1 )*1000, 2 )
    d["k"]=k
    d["labels"]=labels
    
    return d


#d= hdbscan_iteration(loaded_r15, 0.7, 5)



#%%

#%%
#Visualisation de toutes les données
directory = './Bureau/TP-Clustering/artificial/'
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        print(filename)
        load_arff_data(filename)




#%%



def load_txt_data (name) :
    path = './Bureau/TP-Clustering/mysterious-data/'
     
    databrut = pd.read_csv(path+name+".txt", sep=" ", encoding= "ISO-8859-1", skipinitialspace=True)
    
    data = databrut
    datanp = data.to_numpy()

    #Affichage 2D
    f0 = [f[0] for f in datanp]
    f1 = [f[1] for f in datanp]
    plt.scatter(f0,f1,s=8)
    plt.title("Données Mystères")
    plt.show()
    d={}
    d["data"] =datanp
    d["f0"] = f0
    d["f1"] = f1
    return d



#%%
name="complex9"
linkage='single'
distance=12
loaded_data = load_arff_data(name)
agglomerative_iteration(loaded_data, linkage, distance)
linkage='average'
agglomerative_iteration(loaded_data, linkage, distance)
linkage='complete'
agglomerative_iteration(loaded_data, linkage, distance)
#%%
name="triangle1.arff"
loaded_data = load_arff_data(name)
linkage='single'
max_distance=3
agglomerative_evaluation_graph_distance(loaded_data, linkage, max_distance)


#%%


name="cluto-t5-8k.arff"
linkage='single'
max_distance=1.3
loaded_data = load_arff_data(name)
agglomerative_iteration(loaded_data, linkage, distance)
#agglomerative_evaluation_graph_distance(loaded_data, linkage, max_distance)
#neighbors_eps(loaded_data, 4)
#dbscan_iteration(loaded_data, 0.1, 4)
#hdbscan_iteration(loaded_data, 40)
#%%
#Visualisation données mystère
directory = './Bureau/TP-Clustering/mysterious-data/'
i=0
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        print(filename)
        load_txt_data(filename,)
#%%
#Loading mystery data
loaded_dm1 = load_txt_data("x1")
loaded_dm2 = load_txt_data("x2")
loaded_dm3 = load_txt_data("x3")
loaded_dm4 = load_txt_data("x4")
loaded_dm5 = load_txt_data("y1")
loaded_dm6 = load_txt_data("zz1")
loaded_dm7 = load_txt_data("zz2")
#%%
def mystery_data_cluster(loaded_data):
    #Load data
    loaded_data=loaded_dm1
    
    #Determining best parameters
    
    #Clustering with different methods
    kmeans_iteration(loaded_data, k)
    kmedoids_iteration(loaded_data, k)
    agglomerative_iteration(loaded_data, linkage, distance)
    dbscan_iteration(loaded_data, eps, min_samples)
    hdbscan_iteration(loaded_data, min_samples)
    return None
#%%
mystery_data_cluster(loaded_dm1) 