import numpy as np

# ====================================================
#            Unsupervised Learning: KMeans
#     --------------------------------------------
#  benshimol.adir@post.runi.ac.il | 315388850
#  uri.meir@post.runi.ac.il | 206585242
# ====================================================


# KMeans Variants:
# ==================
def kmeans(X, k, p, max_iter=100):
    """
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    centroids = get_random_centroids(X, k)
    centroids, classes = _base_kmeans(X, centroids, p, max_iter)

    return centroids, classes


def kmeans_pp(X, k, p, max_iter=100):
    """
    Your implementation of the kmeans++ algorithm.
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    centroids = get_pp_heuristic_centroids(X, k, p)
    centroids, classes = _base_kmeans(X, centroids, p, max_iter)

    return centroids, classes


def _base_kmeans(X, centroids, p, max_iter):
    """
    Perform the base k-means algorithm.
    """
    EQUALITY_TOLERANCE = 2
    classes = []

    for _ in range(max_iter):
        prev_centroids = centroids.copy()
        classes = _assign_to_classes(X, centroids, p)
        centroids = _recompute_centroids(X, classes, prev_centroids)

        if np.allclose(centroids, prev_centroids, rtol=0, atol=EQUALITY_TOLERANCE):
            break

    return centroids, classes


# Centroids Selection:
# ====================
def get_random_centroids(X, k):
    """
    Each centroid is a point in RGB space (color) in the image.
    This function should uniformly pick `k` centroids from the dataset.
    Input: a single image of shape `(num_pixels, 3)` and `k`, the number of centroids.
    Notice we are flattening the image to a two-dimensional array.
    Output: Randomly chosen centroids of shape `(k,3)` as a numpy array.
    """
    centroid_candidates_indexes = np.random.choice(X.shape[0], size=k, replace=False)
    centroids = X.take(centroid_candidates_indexes, axis=0)

    return np.asarray(centroids).astype(np.float64)


def get_pp_heuristic_centroids(X, k, p):
    """
    Generate the initial centroids using the kmeans++ algorithm.
    """
    initial_centroid, X_rest = _split_chosen_point_from_dataset(X)
    centroids = [initial_centroid]

    for _ in range(1, k):
        weight = _get_weight_probability(X_rest, centroids, p)
        new_centroid, X_rest = _split_chosen_point_from_dataset(X_rest, distribution=weight)
        centroids.append(new_centroid)

    return np.asarray(centroids).astype(np.float64)


#  Metrics:
# ====================
def lp_distance(X, centroids, p=2):
    """
    Inputs:
    A single image of shape (num_pixels, 3)
    The centroids (k, 3)
    The distance parameter p

    output: numpy array of shape `(k, num_pixels)` that's holds the distances of
    all points in RGB space from all centroids
    """
    distances = []

    for centroid in centroids:
        distances.append(np.sum(np.abs(X - centroid) ** p, axis=1) ** (1 / p))

    return np.asarray(distances)


def inertia_metric(X, centroids, p=2):
    """
    Calculate the inertia metric.
    Inertia is the sum of the squared distances between each training instance and its closest centroid.
    """
    return np.sum(np.min(lp_distance(X, centroids, p), axis=0) ** 2)


#  Comparison Test:
# ====================
def run_experiment(algorithm, n_rounds, X, k, p, max_iter=100):
    """
    Run an experiment with the given algorithm and parameters.
    """
    comm_inertia = 0

    for _ in range(n_rounds):
        centroids, _ = algorithm(X, k, p, max_iter)
        comm_inertia += inertia_metric(X, centroids, p)

    return np.mean(comm_inertia)


def performance_comparison(dataset, p, kmeans=kmeans, kmeans_pp=kmeans_pp, n_rounds=10, k_range=range(4, 8),
                           max_iter=100, debug=False):
    """
    Compare the performance of the k-means and k-means++ algorithms on the given dataset.
    """
    # kmeans_results = [run_experiment(algorithm=kmeans, n_rounds=n_rounds, X=dataset, p=p, k=k)
    #                   for k in k_range]
    #
    # kmeans_pp_results = [run_experiment(algorithm=kmeans_pp, n_rounds=n_rounds, X=dataset, p=p, k=k)
    #                      for k in k_range]
    kmeans_results, kmeans_pp_results = [], []

    for k in k_range:
        if debug:
            print(f"test for k = {k}")
            print(f"  [o] test for K-Means")
        kmeans_results.append(
            run_experiment(algorithm=kmeans, n_rounds=n_rounds, X=dataset, p=p, k=k, max_iter=max_iter))
        if debug:
            print(f"  [+] test for K-Means++")
        kmeans_pp_results.append(
            run_experiment(algorithm=kmeans_pp, n_rounds=n_rounds, X=dataset, p=p, k=k, max_iter=max_iter))

    return kmeans_results, kmeans_pp_results, list(k_range)


#  Auxiliary Functions:
# =======================
def _assign_to_classes(X, centroids, p=2):
    """
    Assign each point in the dataset to the closest centroid.
    """
    distances_matrix = lp_distance(X, centroids, p)
    classes = distances_matrix.argmin(axis=0)

    return classes


def _recompute_centroids(X, classes, centroids):
    """
    Recompute the centroids based on the assigned classes.
    """
    recomputed_centroids = np.zeros(centroids.shape)

    for i in range(centroids.shape[0]):
        recomputed_centroids[i, :] = np.mean(X[classes == i], axis=0)

    return recomputed_centroids


def _get_weight_probability(X, centroids, p):
    """
    Calculate the weight probability matrix for the kmeans++ algorithm.
    """
    distances_matrix = lp_distance(X, centroids, p)
    sqrt_distances_matrix = np.min(distances_matrix, axis=0) ** 2
    weight_prob = sqrt_distances_matrix / np.sum(sqrt_distances_matrix)

    return weight_prob


def _choose_random_data_point_index(dataset, distribution=None):
    """
    Choose a random index from the dataset based on the given distribution.
    """
    return np.random.choice(dataset.shape[0], p=distribution)


def _split_chosen_point_from_dataset(dataset, distribution=None):
    """
    Split the chosen data point from the dataset based on the given distribution.
    """
    point_index = _choose_random_data_point_index(dataset, distribution)

    return dataset[point_index], np.delete(dataset, point_index, axis=0)
