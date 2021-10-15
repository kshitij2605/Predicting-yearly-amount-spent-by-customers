import csv
import numpy as np


def import_data():
    X = np.genfromtxt("train_X_lr.csv", delimiter=',', dtype=np.float64, skip_header=1)

    Y = np.genfromtxt("train_Y_lr.csv", delimiter=',', dtype=np.float64)
    return X, Y


def compute_gradient_of_cost_function(X, Y, W):
    #ones = np.ones(len(X))
    #X = np.insert(X, 0, ones, axis=1)

    m = len(X)
    Y_pred = np.dot(X, W)
    difference =  Y_pred - Y
    dW = (1/m) * (np.dot(difference.T, X))
    dW = dW.T
    return dW


def compute_cost(X, Y, W):
    #ones = np.ones(len(X))
    #X = np.insert(X, 0, ones, axis = 1)
    Y_pred = np.dot(X, W)
    difference =  Y_pred - Y                        #h(x) - y
    squared_difference = np.square(difference)      #(h(x) - y)^2
    mse = np.sum(squared_difference)
    cost_value = mse/(2 * len(X))
    return cost_value


def optimize_weights_using_gradient_descent(X, Y, W, num_iterations, learning_rate):
    #ones = np.ones(len(X))
    #X = np.insert(X, 0, ones, axis=1)
    previous_iter_cost = 0
    iter_no = 0
    while True:
        iter_no += 1
        dW = compute_gradient_of_cost_function(X, Y, W)
        W = W - learning_rate * dW
        cost = compute_cost(X, Y, W)
        if iter_no % 10000 == 0:
            print(iter_no, cost)

        if abs(previous_iter_cost - cost) < 0.0000001:
            print(iter_no, cost)
            break

        previous_iter_cost = cost
    return W


def train_model(X, Y):
    X = np.insert(X, 0, 1, axis = 1)
    Y = Y.reshape(len(X), 1)
    W = np.zeros((X.shape[1],1))
    W = optimize_weights_using_gradient_descent(X, Y, W, 10, 0.0002)
    return W


def save_model(weights, weights_file_name):
    with open(weights_file_name, 'w') as weights_file:
        wr = csv.writer(weights_file)
        wr.writerows(weights)
        weights_file.close()


if __name__ == '__main__':
    X, Y = import_data()
    weights = train_model(X, Y)
    save_model(weights, "WEIGHTS_FILE.csv")
