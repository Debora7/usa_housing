import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv('USA_Housing.csv')
x = data[['Avg. Area Income', 'Avg. Area House Age',
          'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms',
          'Area Population']]
y = data[['Price']].values
x_copie = x
y_copie = y
m = len(y)


# Functie de afisare
def plotData(x, y):
    plt.scatter(x, y, 'rx')
    plt.xlabel('Parameters')
    plt.ylabel('Price')


#  Functie pentru separarea datelor
def split(x, t, split):
    indices = np.array(range(len(x)))
    train_size = round(split*len(x))
    random.shuffle(indices)

    train_indices = indices[0:train_size]
    test_indices = indices[train_size:len(x)]

    x_train = x[train_indices, :]
    x_test = x[test_indices, :]
    y_train = y[train_indices, :]
    y_test = y[test_indices, :]

    return x_train, y_train, x_test, y_test


# Functie de normalizare
def normalizare(x, y):
    beta = np.dot((np.linalg.inv(np.dot(x.T, x))), np.dot(x.T, y))

    return beta


# Functie de prezicere
def predict(x_test, beta):
    return np.dot(x_test, beta)


# Functie pentru medie
def metrix(pred, y_test):
    mae = np.mean(np.abs(pred - y_test))
    mse = np.sqare(np.subtract(y_test, pred)).mean()
    rmse = math.sqrt(mse)
    rss = np.sum(np.square(y_test - pred))
    mean = np.mean(y_test)
    sst = np.sum(np.square(y_test - mean))
    r_sq = 1 - (rss / sst)

    return mae, rmse, r_sq


# Functie pentru cost
def cost(x, y, theta):
    eroare = np.dot(x, theta.T) - y
    cost = 1/(2*m) * np.dot(eroare.T, eroare)

    return cost, eroare


# Functie gradient descendent
def grad_desc(x, y, theta, alpha, iteratii):
    cost_vect = []

    for i in range(iteratii):
        cost, eroare = cost(x, y, theta)
        theta = theta - (alpha * (1/m) * np.dot(x.T, eroare))
        cost_vect.append(cost)

    return theta, cost_vect


# Functie de afisare
def afisare_tabel(iteratii, cost_num):
    fix, ax = plt.subplots()
    ax.plot(np.arange(iteratii), cost_num, 'r')
    ax.set_xlabel('Iteratii')
    ax.set_ylabel('Cost')
    ax.set_title('Erori si iteratii')
    plt.show()


def normal(x):
    norm = np.linalg.norm(x)
    norm_vct = x/norm

    return x


one = np.ones((len(x), 1))
x = np.append(one, x, axis=1)
y = np.array(y).reshape((len(y), 1))
print(x.shape)
print(y.shape)

alpha = 0.01
iterations = 1000
split = 0.8
X_train, Y_train, X_test, Y_test = split(x, y, split)

print("TRAINING SET")
print("X_train.shape: ", X_train.shape)
print("Y_train.shape: ", Y_train.shape)

print("TESTING SET")
print("X_test.shape: ", X_test.shape)
print("Y_test.shape: ", Y_test.shape)

X_train = normal(X_train)
X_test = normal(X_test)
beta = normalizare(X_train, Y_train)
predictions = predict(X_test, beta)

print(predictions.shape)

mae, rmse, r_square = metrix(predictions, Y_test)
print("Mean Absolute Error: ", mae)
print("Root Mean Square Error: ", rmse)
print("R square: ", r_square)

X = normal(x_copie)
Y = normal(y_copie)

X = (X - X.mean()) / X.std()
X = np.c_[np.ones(X.shape[0]), X]
theta = np.array([1, 2, -3, 7, 8, -5])

initial_cost, errors = cost(X, Y, theta)
print('With initial theta values of {0}, cost error is {1}'.format(theta, initial_cost))

theta, cost_num = grad_desc(X, Y, theta, alpha, iterations)
afisare_tabel(iterations, cost_num)
final_cost, errors = cost(X, Y, theta)
print('With final theta values of {0}, cost error is {1}'.format(theta, final_cost))
