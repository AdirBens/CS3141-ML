###### Your ID ######
# ID1: 315388850
#####################

# imports 
import numpy as np
import pandas as pd

from itertools import combinations


def preprocess(X, y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """
    X = __normalize_vector(method="mean")(X)
    y = __normalize_vector(method="mean")(y)
    return X, y


def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (m instances over n+1 features).
    """
    X = np.column_stack((np.ones(X.shape[0]), X))
    return X


def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.  

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the cost associated with the current set of parameters (single number).
    """
    J = np.sum(np.square((X @ theta) - y)) / (2 * X.shape[0])
    return J


def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent using 
    the training set. Gradient descent is an optimization algorithm 
    used to minimize some (loss) function by iteratively moving in 
    the direction of steepest descent as defined by the negative of 
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    theta = theta.copy()
    J_history = []

    for _ in range(num_iters):
        J_history.append(compute_cost(X, y, theta))
        theta -= alpha * (((X @ theta) - y) @ X) / X.shape[0]

    return theta, J_history


def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """
    X_t = np.transpose(X)
    pinv = np.linalg.inv(X_t @ X) @ X_t

    return pinv @ y


def efficient_gradient_descent(X, y, theta, alpha, num_iters, threshold=1e-8):
    """
    Learn the parameters of your model using the training set, but stop 
    the learning process once the improvement of the loss value is smaller 
    than 1e-8. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.
    - threshold: The value of improvement between iterations to stop at.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    theta = theta.copy()
    J_history = []

    for _ in range(num_iters):
        J_history.append(compute_cost(X, y, theta))
        if (len(J_history) > 1) and ((J_history[-2] - J_history[-1]) < threshold):
            break
        else:
            theta -= alpha * (((X @ theta) - y) @ X) / X.shape[0]
    return theta, J_history


def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of alpha and train a model using 
    the training dataset. maintain a python dictionary with alpha as the 
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - alpha_dict: A python dictionary - {alpha_value : validation_loss}
    """
    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {}
    initial_theta = __guess_init_theta(X_train.shape[1])

    for alpha in alphas:
        theta, _ = efficient_gradient_descent(X_train, y_train, initial_theta, alpha, iterations)
        alpha_dict[alpha] = compute_cost(X_val, y_val, theta)

    return alpha_dict


def forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations, n_selected=5):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features, potentially reducing overfitting, 
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_alpha: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent
    - n_selected: the desired number of selected features

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    X_train, X_val = apply_bias_trick(X_train), apply_bias_trick(X_val)

    selected_features = [0]  # insert bias column to selected features
    candidates_features = list(range(1, X_train.shape[1]))

    for _ in range(n_selected):
        J_history = dict()

        for feature in candidates_features:
            selected_features.append(feature)
            initial_theta = __guess_init_theta(X_train[:, selected_features].shape[1])
            theta, _ = efficient_gradient_descent(X_train[:, selected_features], y_train,
                                                  initial_theta, best_alpha, iterations)
            J_history[feature] = compute_cost(X_val[:, selected_features], y_val, theta)
            selected_features.pop()

        best_feature = min(J_history, key=J_history.get)
        candidates_features.remove(best_feature)
        selected_features.append(best_feature)

    selected_features = [i - 1 for i in selected_features if i != 0]  # remove bias 'feature' and fix indexes.
    return selected_features


def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (m instances over n features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """
    df_poly = df.copy()

    # create all combination between two different columns
    df_poly = pd.concat([df_poly, pd.DataFrame({f"{c_a} * {c_b}": df[c_a] * df[c_b]
                                                for c_a, c_b in combinations(df.columns, 2)})], axis=1)

    # create self columns product
    df_poly = pd.concat([df_poly, df.copy().add_suffix("^2") ** 2], axis=1)
    return df_poly


#######################
# Auxiliary functions #
#######################
def __normalize_vector(method="mean"):
    """
    Feature scaling is a method used to normalize the range of independent variables or features of data.

    Input:
    - method: [default is Mean normalization] The scaling method to use.

    Returns:
    - A function that apply the scaling method on a given vector (numpy array like)
    """
    methods = {
        "mean": lambda v: np.divide(v - v.mean(axis=0), (v.max(axis=0) - v.min(axis=0)))
    }
    return methods.get(method)


def __guess_init_theta(shape):
    """
    Guess initial value for theta vector
    Input:
    - shape: the size of the vector

    Returns:
    - vector of size (shape, ) with guessed theta values
    """
    np.random.seed(42)
    return np.random.random((shape,))
