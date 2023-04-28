import numpy as np
import matplotlib.pyplot as plt

### Chi square table values ###
# The first key is the degree of freedom
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5: 0.45,
                 0.25: 1.32,
                 0.1: 2.71,
                 0.05: 3.84,
                 0.0001: 100000},
             2: {0.5: 1.39,
                 0.25: 2.77,
                 0.1: 4.60,
                 0.05: 5.99,
                 0.0001: 100000},
             3: {0.5: 2.37,
                 0.25: 4.11,
                 0.1: 6.25,
                 0.05: 7.82,
                 0.0001: 100000},
             4: {0.5: 3.36,
                 0.25: 5.38,
                 0.1: 7.78,
                 0.05: 9.49,
                 0.0001: 100000},
             5: {0.5: 4.35,
                 0.25: 6.63,
                 0.1: 9.24,
                 0.05: 11.07,
                 0.0001: 100000},
             6: {0.5: 5.35,
                 0.25: 7.84,
                 0.1: 10.64,
                 0.05: 12.59,
                 0.0001: 100000},
             7: {0.5: 6.35,
                 0.25: 9.04,
                 0.1: 12.01,
                 0.05: 14.07,
                 0.0001: 100000},
             8: {0.5: 7.34,
                 0.25: 10.22,
                 0.1: 13.36,
                 0.05: 15.51,
                 0.0001: 100000},
             9: {0.5: 8.34,
                 0.25: 11.39,
                 0.1: 14.68,
                 0.05: 16.92,
                 0.0001: 100000},
             10: {0.5: 9.34,
                  0.25: 12.55,
                  0.1: 15.99,
                  0.05: 18.31,
                  0.0001: 100000},
             11: {0.5: 10.34,
                  0.25: 13.7,
                  0.1: 17.27,
                  0.05: 19.68,
                  0.0001: 100000}}


def calc_gini(data) -> float:
    """
    Calculate gini impurity measure of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - gini: The gini impurity value.
    """
    _, groups_size = _count_by_col(data, -1)
    data_size = data.shape[0]

    return 1 - np.sum((groups_size / data_size) ** 2)


def calc_entropy(data) -> float:
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    _, groups_size = _count_by_col(data, -1)
    data_size = data.shape[0]

    return -1 * np.sum((groups_size / data_size) * np.log2(groups_size / data_size))


def goodness_of_split(data, feature, impurity_func, gain_ratio=False) -> tuple[float, dict]:
    """
    Calculate the goodness of split of a dataset given a feature and impurity function.
    Note: Python support passing a function as arguments to another function
    Input:
    - data: any dataset where the last column holds the labels.
    - feature: the feature index the split is being evaluated according to.
    - impurity_func: a function that calculates the impurity.
    - gain_ratio: goodness of split or gain ratio flag.

    Returns:
    - goodness: the goodness of split value
    - groups: a dictionary holding the data after splitting
              according to the feature values.
    """
    feature_values, _ = np.unique(_fetch_column(data, feature), return_counts=True)
    groups = {value: data[_fetch_column(data, feature) == value] for value in feature_values}
    split_information = 0

    if gain_ratio is True:
        impurity_func = calc_entropy
        split_information = _calc_split_information(data, groups)

    goodness = impurity_func(data) - np.sum([(groups[subset].shape[0] / data.shape[0]) * impurity_func(groups[subset])
                                             for subset in groups.keys()])

    if gain_ratio is True and split_information != 0:
        goodness /= split_information

    return goodness, groups


class DecisionNode:

    def __init__(self, data, feature=-1, depth=0, chi=1, max_depth=1000, gain_ratio=False):

        self.data = data  # the relevant data for the node
        self.feature = feature  # column index of criteria being tested
        self.pred = self.calc_node_pred()  # the prediction of the node
        self.depth = depth  # the current depth of the node
        self.children = []  # array that holds this nodes children
        self.children_values = []
        self.terminal = True  # determines if the node is a leaf
        self.chi = chi
        self.max_depth = max_depth  # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio

    def calc_node_pred(self) -> str:
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        labels, groups_size = _count_by_col(self.data, -1)
        labels_groups = dict(zip(labels, groups_size))

        return max(labels_groups, key=labels_groups.get)

    def add_child(self, node, val) -> None:
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        self.children.append(node)
        self.children_values.append(val)

    def split(self, impurity_func) -> None:
        """
        Splits the current node according to the impurity_func. This function finds
        the best feature to split according to, and create the corresponding children.
        This function should support pruning according to chi and max_depth.

        Input:
        - The impurity function that should be used as the splitting criteria

        This function has no return value
        """
        self.feature = _find_best_feature(self.data, impurity_func, self.gain_ratio)
        _, children_groups = goodness_of_split(self.data, self.feature, impurity_func, self.gain_ratio)

        if len(children_groups) < 2:
            self.terminal = True
            return

        # create children
        for children_value, group in children_groups.items():
            new_child = DecisionNode(data=group, feature=-1, depth=self.depth + 1, chi=self.chi,
                                     max_depth=self.max_depth, gain_ratio=self.gain_ratio)
            self.add_child(new_child, children_value)

    def chi_square_test(self) -> float:
        """
        Calculate the chi square value of this node's split

        Output:
        - chi_square value (float)
        """
        labels, groups_sizes = _count_by_col(self.data, -1)
        labels_probability = dict(zip(labels, np.divide(groups_sizes, self.data.shape[0])))
        chi_square = 0

        for child in self.children:
            child_labels, child_groups_sizes = _count_by_col(child.data, -1)
            child_labels_sizes = dict(zip(child_labels, child_groups_sizes))

            for label in labels_probability:
                E_label = labels_probability[label] * child.data.shape[0]
                n_f = child_labels_sizes.get(label, 0)
                chi_square += ((n_f - E_label) ** 2) / E_label

        return chi_square

    def is_split_randomly(self) -> bool:
        """
        Determine if this node was split in random matter, Using `Chi Square Test` and chi_table lookup.

        Output:
        - True if random, False else.
        """
        freedom_deg = len(self.children_values) - 1
        if (freedom_deg < 1) or (self.chi == 1):
            return False

        return self.chi_square_test() < chi_table[freedom_deg][self.chi]


def build_tree(data, impurity, gain_ratio=False, chi=1, max_depth=1000) -> DecisionNode:
    """
    Build a tree using the given impurity measure and training dataset.
    You are required to fully grow the tree until all leaves are pure unless
    you are using pruning

    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.
    - gain_ratio: goodness of split or gain ratio flag

    Output: the root node of the tree.
    """
    root = DecisionNode(data=data, depth=0, chi=chi, max_depth=max_depth, gain_ratio=gain_ratio)
    _build_tree(root, impurity)

    return root


def _build_tree(node, impurity) -> None:
    """
    Recursive call of the driver method build_tree
    @see build_tree

    Input:
    - node: a DecisionNode instance rooted the subtree
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.
    """
    if (node.depth == node.max_depth) or (impurity(node.data) == 0):
        return

    # Decide if split current node, using `Chi Square Test`
    node.split(impurity)
    if not node.is_split_randomly():
        for child in node.children:
            _build_tree(child, impurity)

        if len(node.children) != 0:
            node.terminal = False


def predict(root, instance) -> str:
    """
    Predict a given instance using the decision tree

    Input:
    - root: the root of the decision tree.
    - instance: an row vector from the dataset. Note that the last element
                of this vector is the label of the instance.

    Output: the prediction of the instance.
    """
    node = root
    while node.terminal is False:
        attribute = instance[node.feature]
        try:
            node = node.children[node.children_values.index(attribute)]
        except ValueError:
            break

    return node.pred


def calc_accuracy(node, dataset) -> float:
    """
    Predict a given dataset using the decision tree and calculate the accuracy

    Input:
    - node: a node in the decision tree.
    - dataset: the dataset on which the accuracy is evaluated

    Output: the accuracy of the decision tree on the given dataset (%).
    """
    accuracy = sum(predict(node, row) == row[-1] for row in dataset)

    return 100 * (accuracy / dataset.shape[0])


def depth_pruning(X_train, X_test) -> tuple[list[float], list[float]]:
    """
    Calculate the training and testing accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously.

    Input:
    - X_train: the training data where the last column holds the labels
    - X_test: the testing data where the last column holds the labels

    Output: the training and testing accuracies per max depth
    """
    IMPURITY_FUNCTION = calc_entropy
    GAIN_RATIO = True

    training = []
    testing = []

    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        root = build_tree(data=X_train, impurity=IMPURITY_FUNCTION, gain_ratio=GAIN_RATIO, max_depth=max_depth)
        training.append(calc_accuracy(root, X_train))
        testing.append(calc_accuracy(root, X_test))

    return training, testing


def chi_pruning(X_train, X_test) -> tuple[list[float], list[float], list[int]]:
    """
    Calculate the training and testing accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously.

    Input:
    - X_train: the training data where the last column holds the labels
    - X_test: the testing data where the last column holds the labels

    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_testing_acc: the testing accuracy per chi value
    - depths: the tree depth for each chi value
    """
    IMPURITY_FUNCTION = calc_entropy
    GAIN_RATIO = True

    chi_training_acc = []
    chi_testing_acc = []
    depth = []

    for chi_val in [1, 0.5, 0.25, 0.1, 0.05, 0.0001]:
        root = build_tree(X_train, IMPURITY_FUNCTION, GAIN_RATIO, chi=chi_val)
        chi_training_acc.append(calc_accuracy(root, X_train))
        chi_testing_acc.append(calc_accuracy(root, X_test))
        depth.append(_get_tree_depth(root))

    return chi_training_acc, chi_testing_acc, depth


def count_nodes(node) -> int:
    """
    Input:
    - node: DecisionNode, usually the root of a DecisionNodes Tree

    Output:
    - The number of nodes in the tree rooted by the given node (int)
    """
    if node.terminal is True:
        return 1

    return 1 + sum([count_nodes(child) for child in node.children])


#######################
# AUXILIARY Functions #
#######################

def _fetch_column(data, column_index: int = -1) -> np.array:
    """
    Fetch column by index from given data-set (usually np.array)

    Input:
    - data: data-set (usually np.array)
    - column_index: the index of the desire column (int)

    Output:
    - the desire column (as np.array)
    """
    return data[:, column_index]


def _count_by_col(data, column_index: int = -1) -> tuple[str, np.array]:
    """
    given data-set column, count rows group by unique labels.

    Input:
    - data: data-set (usually np.array)
    - column_index: the index of the desire column (int)

    Output:
    - labels: list of unique labels (list)
    - group_sizes: the size of group corresponds to labels (np.array)
    """
    labels, groups_size = np.unique(_fetch_column(data, column_index), return_counts=True)
    return labels, groups_size


def _calc_split_information(data, groups) -> float:
    """
    Calculate the split_information value of a given data and groups (partiotion of data)

    Input:
    - data: data-set (usually np.array)
    - groups: dict holds partition of the data into groups.

    Output:
    - split_information value (float)
    """
    split_information = -1 * np.sum((groups[subset].shape[0] / data.shape[0]) *
                                    np.log2(groups[subset].shape[0] / data.shape[0])
                                    for subset in groups.keys())

    return split_information


def _find_best_feature(data, impurity_func, gain_ratio) -> int:
    """
    Find the best feature (highest gain score) to split by the data, calculate according to given impurity function.

    Input:
    - data: data-set (usually np.array)
    - impurity_func: a function that calculates the impurity.
    - gain_ratio: goodness of split or gain ratio flag.

    Output:
    - best_feature - the index of the best feature to split the data (int)
    """
    feature_candidates = range(data.shape[1] - 1)
    gain = {feature: goodness_of_split(data, feature, impurity_func, gain_ratio)[0]
            for feature in feature_candidates}

    best_feature = max(gain, key=gain.get)
    return best_feature


def _get_tree_depth(root) -> int:
    """
    Traverse the tree rooted by root and Calculate its depth.

    Input:
    - root: DecisionNode rooted the tree

    Output:
    - The tree depth (int)
    """
    if root.terminal is True:
        return root.depth
    else:
        return max([_get_tree_depth(child) for child in root.children])
