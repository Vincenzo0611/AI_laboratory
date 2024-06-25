import numpy as np
import random

def initialize_centroids_forgy(data, k):
    return random.choices(data, k=k)

def initialize_centroids_kmeans_pp(data, k):
    centroids = [random.choice(data)]
    while len(centroids) < k:
        distances = np.array([min([np.linalg.norm(np.array(x) - np.array(c)) ** 2 for c in centroids]) for x in data])
        farthest_idx = np.argmax(distances)
        centroids.append(data[farthest_idx])
    return centroids

def assign_to_cluster(data, centroid):
    assignments = []
    for point in data:
        distances = [np.linalg.norm(np.array(point) - np.array(c)) for c in centroid]
        cluster_index = np.argmin(distances)
        assignments.append(cluster_index)
    return assignments

def update_centroids(data, assignments):
    new_centroids = []
    num_clusters = max(assignments) + 1

    for cluster_idx in range(num_clusters):
        cluster_points = [data[i] for i, c in enumerate(assignments) if c == cluster_idx]
        if len(cluster_points) > 0:
            cluster_mean = np.mean(cluster_points, axis=0)
            new_centroids.append(cluster_mean)
        else:
            new_centroids.append(data[cluster_idx])

    return new_centroids


def mean_intra_distance(data, assignments, centroids):
    assignments = np.array(assignments)
    return np.sqrt(np.sum((data - np.array(centroids)[assignments]) ** 2))


def k_means(data, num_centroids, kmeansplusplus):
    # centroids initizalization
    if kmeansplusplus:
        centroids = initialize_centroids_kmeans_pp(data, num_centroids)
    else: 
        centroids = initialize_centroids_forgy(data, num_centroids)

    assignments = assign_to_cluster(data, centroids)
    for i in range(100): # max number of iteration = 100
        #print(f"Intra distance after {i} iterations: {mean_intra_distance(data, assignments, centroids)}")
        centroids = update_centroids(data, assignments)
        new_assignments = assign_to_cluster(data, centroids)
        if np.all(new_assignments == assignments): # stop if nothing changed
            break
        else:
            assignments = new_assignments

    return new_assignments, centroids, mean_intra_distance(data, new_assignments, centroids)         

