import time
import warnings
import numpy as np
import myplots as myplt
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.datasets import make_blobs,make_circles,make_moons
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
import scipy.io as io
from matplotlib.backends.backend_pdf import PdfPages
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster  #

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

"""
Part 4.	
Evaluation of Hierarchical Clustering over Diverse Datasets:
In this task, you will explore hierarchical clustering over different datasets. You will also evaluate different ways to merge clusters and good ways to find the cut-off point for breaking the dendrogram.
"""

# Fill these two functions with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_hierarchical_cluster(data, linkage_type, n_clusters):
    model = AgglomerativeClustering(linkage=linkage_type, n_clusters=n_clusters)
    # Standardize the data
    scaler = StandardScaler()
    data_std = scaler.fit_transform(data)

    # Train AgglomerativeClustering model
    #model = AgglomerativeClustering(linkage=linkage_type, n_clusters=n_clusters)
    model.fit(data_std)

    # Return the label predictions
    return model.labels_



def fit_modified(data, linkage_method, elbow_threshold=0.05):
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Compute the linkage matrix
    Z = linkage(data_scaled, method=linkage_method)

    # Calculate the distances between each merge
    distances = Z[:, 2]
    # Calculate the rate of distance increase between successive merges
    distance_diffs = np.diff(distances)
    # Normalize by the max distance to avoid scale issues
    normalized_distance_diffs = distance_diffs / distances.max()

    # Find the "elbow" in the distances, which may be a good cutoff for clustering
    elbow_index = np.where(normalized_distance_diffs > elbow_threshold)[0]
    if len(elbow_index) > 0:
        cutoff_distance = distances[elbow_index[0]]
    else:
        cutoff_distance = distances[-1]  # No elbow found, use the last merge distance

    # Perform clustering with the determined cutoff distance
    model = AgglomerativeClustering(
        n_clusters=None, 
        distance_threshold=cutoff_distance, 
        linkage=linkage_method
    )
    model.fit(data_scaled)
    return model.labels_

def compute():
    answers = {}

    """
    A.	Repeat parts 1.A and 1.B with hierarchical clustering. That is, write a function called fit_hierarchical_cluster (or something similar) that takes the dataset, the linkage type and the number of clusters, that trains an AgglomerativeClustering sklearn estimator and returns the label predictions. Apply the same standardization as in part 1.B. Use the default distance metric (euclidean) and the default linkage (ward).
    """
    n_samples = 100
    random_state = 42

# Generate the datasets
    nc = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=random_state)
    nm = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=random_state)
    b = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    bvv = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)

    # Anisotropicly distributed data
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    add = (X_aniso, y)
    

    # Dictionary of 5 datasets. e.g., dct["nc"] = [data, labels]
    # keys: 'nc', 'nm', 'bvv', 'add', 'b' (abbreviated datasets)
    dct = answers["4A: datasets"] = {'nc': [nc[0],nc[1]],
                                     'nm': [nm[0],nm[1]],
                                     'bvv': [bvv[0],bvv[1]],
                                     'add': [add[0],add[1]],
                                     'b': [b[0],b[1]]}

    # dct value:  the `fit_hierarchical_cluster` function
    dct = answers["4A: fit_hierarchical_cluster"] = fit_hierarchical_cluster

    """
    B.	Apply your function from 4.A and make a plot similar to 1.C with the four linkage types (single, complete, ward, centroid: rows in the figure), and use 2 clusters for all runs. Compare the results to problem 1, specifically, are there any datasets that are now correctly clustered that k-means could not handle?

    Create a pdf of the plots and return in your report. 
    
    """
    linkage_methods = ['single', 'complete', 'ward', 'average']  # using 'average' instead of 'centroid'
    hierarchical_results = {}

# Preparing the structure to store results for each linkage method and dataset
    for dataset_name, (X, y) in answers["4A: datasets"].items():
        hierarchical_results[dataset_name] = {}
        for linkage_method in linkage_methods:
            labels = fit_hierarchical_cluster(X, linkage_method, 2)
            hierarchical_results[dataset_name][linkage_method] = labels

# Adapt the kmeans_dct dictionary to include all linkage methods for each dataset
    kmeans_dct = {}
    for dataset_name in hierarchical_results:
        X, y = answers["4A: datasets"][dataset_name]
    # Create a nested dictionary for each dataset that contains labels for each linkage method
        kmeans_dct[dataset_name] = ((X, y), hierarchical_results[dataset_name])


    myplt.plot_part1C(kmeans_dct, 'part4_b.jpg')

# dct value: list of dataset abbreviations (see 1.C)
    
    dct = answers["4B: cluster successes"] = ["nc","nm"]

    """
    C.	There are essentially two main ways to find the cut-off point for breaking the diagram: specifying the number of clusters and specifying a maximum distance. The latter is challenging to optimize for without knowing and/or directly visualizing the dendrogram, however, sometimes simple heuristics can work well. The main idea is that since the merging of big clusters usually happens when distances increase, we can assume that a large distance change between clusters means that they should stay distinct. Modify the function from part 1.A to calculate a cut-off distance before classification. Specifically, estimate the cut-off distance as the maximum rate of change of the distance between successive cluster merges (you can use the scipy.hierarchy.linkage function to calculate the linkage matrix with distances). Apply this technique to all the datasets and make a plot similar to part 4.B.
    
    Create a pdf of the plots and return in your report. 
    """
        # dct is the function described above in 4.C
    dct = answers["4A: modified function"] = fit_modified

    linkage_methods = ['single', 'complete', 'ward', 'average']  # using 'average' instead of 'centroid'
    modified_results = {}

# Standardize and compute modified clustering for each dataset and linkage method
    for dataset_name, (X, y) in answers["4A: datasets"].items():
        modified_results[dataset_name] = {}
        for linkage_method in linkage_methods:
            labels = fit_modified(X, linkage_method)
            modified_results[dataset_name][linkage_method] = labels

# Adapt the structure to fit myplots.plot_part1C
    plot_dct = {}
    for dataset_name in modified_results:
        X, y = answers["4A: datasets"][dataset_name]
        plot_dct[dataset_name] = ((X, y), modified_results[dataset_name])
    myplt.plot_part1C(plot_dct, "part4_c.jpg")

    

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part4.pkl", "wb") as f:
        pickle.dump(answers, f)
