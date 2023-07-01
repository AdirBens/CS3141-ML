import numpy as np


def get_random_centroids(X, k):
    """
    Each centroid is a point in RGB space (color) in the image.
    This function should uniformly pick `k` centroids from the dataset.
    Input: a single image of shape `(num_pixels, 3)` and `k`, the number of centroids.
    Notice we are flattening the image to a two-dimensional array.
    Output: Randomly chosen centroids of shape `(k,3)` as a numpy array.
    """
    _init_random_generator()
    centroid_candidates_indexes = np.random.choice(X.shape[0], size=k, replace=False)
    centroids = X.take(centroid_candidates_indexes, axis=0)

    return np.asarray(centroids).astype(np.float64)


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


def performance_comparison(dataset, p, kmeans=kmeans, kmeans_pp=kmeans_pp, n_rounds=10, k_range=range(4, 16)):
    kmeans_results = [run_experiment(algorithm=kmeans, n_rounds=n_rounds, X=dataset, p=p, k=k)
                      for k in k_range]

    kmeans_pp_results = [run_experiment(algorithm=kmeans_pp, n_rounds=n_rounds, X=dataset, p=p, k=k)
                         for k in k_range]

    return kmeans_results, kmeans_pp_results, list(k_range)


def run_experiment(algorithm, n_rounds, X, k, p, max_iter=100):
    comm_inertia = 0

    for _ in range(n_rounds):
        centroids, _ = algorithm(X, k, p, max_iter)
        comm_inertia += inertia_metric(X, centroids, p)

    return np.mean(comm_inertia)


def inertia_metric(X, centroids, p=2):
    """
    calculate the inertia metric.
    inertia is the sum of the squared distances between each training instance and its closest centroid
    """
    return np.sum(np.min(lp_distance(X, centroids, p), axis=0) ** 2)


def get_pp_heuristic_centroids(X, k, p):
    initial_centroid, X_rest = _split_chosen_point_from_dataset(X)
    centroids = [initial_centroid]

    for _ in range(1, k):
        weight = _get_weight_probability(X_rest, centroids, p)
        new_centroid, X_rest = _split_chosen_point_from_dataset(X_rest, distribution=weight)
        centroids.append(new_centroid)

    return np.asarray(centroids).astype(np.float64)


def _init_random_generator(random_state: int = 42):
    np.random.seed(seed=random_state)


def _base_kmeans(X, centroids, p, max_iter):
    classes = []

    for _ in range(max_iter):
        classes = _assign_to_classes(X, centroids, p)
        prev_centroids = centroids
        centroids = _recompute_centroids(X, classes, prev_centroids)

        if np.array_equal(centroids, prev_centroids):
            break

    return centroids, classes


def _assign_to_classes(X, centroids, p=2):
    distances_matrix = lp_distance(X, centroids, p)
    classes = distances_matrix.argmin(axis=0)

    return classes


def _recompute_centroids(X, classes, centroids):
    recomputed_centroids = np.zeros(centroids.shape)

    for i in range(centroids.shape[0]):
        recomputed_centroids[i, :] = np.mean(X[classes == i], axis=0)

    return recomputed_centroids


def _get_weight_probability(X, centroids, p):
    distances_matrix = lp_distance(X, centroids, p)
    sqrt_distances_matrix = np.min(distances_matrix, axis=0) ** 2
    weight_prob_matrix = sqrt_distances_matrix / np.sum(sqrt_distances_matrix)

    return weight_prob_matrix


def _choose_random_data_point_index(dataset, distribution=None):
    _init_random_generator()
    return np.random.choice(dataset.shape[0], p=distribution)


def _split_chosen_point_from_dataset(dataset, distribution=None):
    point_index = _choose_random_data_point_index(dataset, distribution)

    return dataset[point_index], np.delete(dataset, point_index, axis=0)
