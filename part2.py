from pprint import pprint

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

import myplots as myplt
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
import scipy.io as io
from scipy.cluster.hierarchy import dendrogram, linkage  #

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

# ----------------------------------------------------------------------
"""
Part 2
Comparison of Clustering Evaluation Metrics: 
In this task you will explore different methods to find a good value for k
"""

# Fill this function with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_kmeans(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data)
    # SSE is the sum of squared distances of samples to their closest cluster center
    distances = np.sqrt(np.sum((data - kmeans.cluster_centers_[kmeans.labels_])**2,axis=1))
    sse=np.sum(distances**2)
    return sse, kmeans.inertia_




def compute():
    # ---------------------
    answers = {}

    """
    A.	Call the make_blobs function with following parameters :(center_box=(-20,20), n_samples=20, centers=5, random_state=12).
    """
    n_samples=20
    centers=5 
    center_box=(-20, 20) 
    random_state=12
    X,label=datasets.make_blobs(n_samples=n_samples,centers=centers,center_box=center_box,random_state=random_state)
    co_1=X[0:,0:1]
    co_2=X[0:,1:]



    # dct: return value from the make_blobs function in sklearn, expressed as a list of three numpy arrays
    dct = answers["2A: blob"] = [co_1,co_2,label]
    #print(dct)
    """
    B. Modify the fit_kmeans function to return the SSE (see Equations 8.1 and 8.2 in the book).
    """
    # dct value: the `fit_kmeans` function
    dct = answers["2B: fit_kmeans"] = fit_kmeans
    print(dct)
    """
    C.	Plot the SSE as a function of k for k=1,2,….,8, and choose the optimal k based on the elbow method.
    """
    
    sse_values = []
    for k in range(1, 9):
        sse, inertia = fit_kmeans(X, k)
        sse_values.append([k, sse])

    
    # dct value: a list of tuples, e.g., [[0, 100.], [1, 200.]]
    # Each tuple is a (k, SSE) pair
    dct = answers["2C: SSE plot"] = sse_values
    print(sse_values)
    """
    sse_values = answers["2C: SSE plot"]
    k_values, sse_scores = zip(*sse_values)
    fig=plt.figure(figsize=(20,10))
    plt.plot(k_values, sse_scores, 'o-', linewidth=2, markersize=10)
    plt.title('SSE vs. Number of Clusters')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('SSE (Sum of Squared Errors)')
    plt.xticks(range(1, 9))  # Ensure x-ticks for each k value
    plt.grid(False)

    plt.savefig('part2)ques_c.jpg')
    """
   
    """
    D.	Repeat part 2.C for inertia (note this is an attribute in the kmeans estimator called _inertia). Do the optimal k’s agree?
    """
    inertia_values = []
    for k in range(1, 9):
        _, inertia = fit_kmeans(X, k)  # Extract only the inertia value from the function's return
        inertia_values.append([k, inertia])
    # dct value has the same structure as in 2C
    dct = answers["2D: inertia plot"] =inertia_values
    """
    inertia_values = answers["2D: inertia plot"]
    k_values, inertia_scores = zip(*inertia_values)
    fig=plt.figure(figsize=(20,10))
    plt.plot(k_values, inertia_scores, 'o-', linewidth=2, markersize=10)
    plt.title('inertia vs. Number of Clusters')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('inertia ')
    plt.xticks(range(1, 9))  # Ensure x-ticks for each k value
    plt.grid(False)
    plt.savefig('part2)ques_d.jpg')
   
    #print(dct)
    """
    
    # dct value should be a string, e.g., "yes" or "no"
    dct = answers["2D: do ks agree?"] = "no"

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part2.pkl", "wb") as f:
        pickle.dump(answers, f)
