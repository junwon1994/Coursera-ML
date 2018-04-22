#  Machine Learning Online Class
#  Exercise 8 | Anomaly Detection and Collaborative Filtering
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     estimateGaussian.py
#     selectThreshold.py
#     cofiCostFunc.py
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.

import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy.optimize import minimize

from cofiCostFunc import cofiCostFunc
from checkCostFunction import checkCostFunction
from loadMovieList import loadMovieList
from normalizeRatings import normalizeRatings

#  =============== Part 1: Loading movie ratings dataset ================
#  You will start by loading the movie ratings dataset to understand the
#  structure of the data.

print('Loading movie ratings dataset.\n')

# Load data
data = loadmat('ex8_movies.mat')
Y = data['Y']
R = data['R']

# Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on
# 943 users
#
# R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
# rating to movie i

# From the matrix, we can compute statistics like average rating.
print('Average rating for movie 1 (Toy Story): {:.2f} / 5\n'.format(
    np.mean(Y[0, R[0]])))

# We can "visualize" the ratings matrix by plotting it with pyplot.imshow
plt.imshow(
    Y,
    aspect='equal',
    origin='upper',
    extent=(0, Y.shape[1], 0, Y.shape[0] / 2))
plt.ylabel('Movies')
plt.xlabel('Users')
plt.show(block=False)

input('\nProgram paused. Press Enter to continue.\n')

#  ============ Part 2: Collaborative Filtering Cost Function ===========
#  You will now implement the cost function for collaborative filtering.
#  To help you debug your cost function, we have included set of weights
#  that we trained on that. Specifically, you should complete the code in
#  cofiCostFunc.m to return J.

# Load pre-trained weights (X, Theta, num_users, num_movies, num_features)
data = loadmat('ex8_movieParams.mat')
X = data['X']
Theta = data['Theta']
n_users = np.asscalar(data['num_users'])
n_movies = np.asscalar(data['num_movies'])
n_features = np.asscalar(data['num_features'])

# Reduce the data set size so that this runs faster
n_users = 4
n_movies = 5
n_features = 3
X = X[:n_movies, :n_features]
Theta = Theta[:n_users, :n_features]
Y = Y[:n_movies, :n_users]
R = R[:n_movies, :n_users]

# Evaluate cost function
J, _ = cofiCostFunc(
    np.r_[X.ravel(order='F'), Theta.ravel(order='F')],
    Y,
    R,
    n_users,
    n_movies,
    n_features,
    0)

print('Cost at loaded parameters: {:.2f} \n'
      '(this value should be about 22.22)'.format(J))

input('\nProgram paused. Press Enter to continue.\n')

#  ============== Part 3: Collaborative Filtering Gradient ==============
#  Once your cost function matches up with ours, you should now implement
#  the collaborative filtering gradient function. Specifically, you should
#  complete the code in cofiCostFunc.m to return the grad argument.
#
print('\nChecking Gradients (without regularization) ... ')

# Check gradients by running checkNNGradients
checkCostFunction()

input('\nProgram paused. Press Enter to continue.\n')

#  ========= Part 4: Collaborative Filtering Cost Regularization ========
#  Now, you should implement regularization for the cost function for
#  collaborative filtering. You can implement it by adding the cost of
#  regularization to the original cost computation.

#  Evaluate cost function
J, _ = cofiCostFunc(
    np.r_[X.ravel(order='F'), Theta.ravel(order='F')],
    Y,
    R,
    n_users,
    n_movies,
    n_features,
    1.5)

print('Cost at loaded parameters (lambda = 1.5): {:.2f} '
      '\n(this value should be about 31.34)'.format(J))

input('\nProgram paused. Press Enter to continue.\n')

#  ======= Part 5: Collaborative Filtering Gradient Regularization ======
#  Once your cost matches up with ours, you should proceed to implement
#  regularization for the gradient.
#

#
print('\nChecking Gradients (with regularization) ... ')

#  Check gradients by running checkNNGradients
checkCostFunction(1.5)

input('\nProgram paused. Press Enter to continue.\n')

#  ============== Part 6: Entering ratings for a new user ===============
#  Before we will train the collaborative filtering model, we will first
#  add ratings that correspond to a new user that we just observed. This
#  part of the code will also allow you to put in your own ratings for the
#  movies in our dataset!
#
movieList = loadMovieList()

# Initialize my ratings
my_ratings = np.zeros(1682)

# Check the file movie_idx.txt for id of each movie in our dataset
# For example, Toy Story (1995) has ID 1, so to rate it "4", you can set
my_ratings[0] = 4

# Or suppose did not enjoy Silence of the Lambs (1991), you can set
my_ratings[97] = 2

# We have selected a few movies we liked / did not like and the ratings we
# gave are as follows:
my_ratings[6] = 3
my_ratings[11] = 5
my_ratings[53] = 4
my_ratings[63] = 5
my_ratings[65] = 3
my_ratings[68] = 5
my_ratings[182] = 4
my_ratings[225] = 5
my_ratings[354] = 5

print('\n\nNew user ratings:')
for i, my_rating in enumerate(my_ratings):
    if my_rating > 0:
        print('Rated {:d} for {:s}'.format(int(my_rating), movieList[i]))

input('\nProgram paused. Press Enter to continue.\n')

#  ================== Part 7: Learning Movie Ratings ====================
#  Now, you will train the collaborative filtering model on a movie rating
#  dataset of 1682 movies and 943 users
#

print('\nTraining collaborative filtering...')

#  Load data
data = loadmat('ex8_movies.mat')
Y = data['Y']
R = data['R']

#  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies by
#  943 users
#
#  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
#  rating to movie i

#  Add our own ratings to the data matrix
Y = np.c_[my_ratings, Y]
R = np.c_[my_ratings != 0, R]

#  Normalize Ratings
Ynorm, Ymean = normalizeRatings(Y, R)

#  Useful Values
n_users = np.size(Y, 1)
n_movies = np.size(Y, 0)
n_features = 10

# Set Initial Parameters (Theta, X)
X = np.random.randn(n_movies, n_features)
Theta = np.random.randn(n_users, n_features)

initial_parameters = np.r_[X.ravel(order='F'), Theta.ravel(order='F')]

# Set options for minimize
options = {'disp': True, 'maxiter': 1000}

# Set Regularization
Lambda = 10
theta = minimize(
    lambda t: cofiCostFunc(t, Ynorm, R, n_users, n_movies, n_features, Lambda),
    initial_parameters,
    method='CG',
    jac=True,
    options=options)['x']

# Unfold the returned theta back into U and W
X = np.reshape(
    theta[:n_movies * n_features], (n_movies, n_features), order='F')
Theta = np.reshape(
    theta[n_movies * n_features:], (n_users, n_features), order='F')

print('Recommender system learning completed.')

input('\nProgram paused. Press Enter to continue.\n')

#  ================== Part 8: Recommendation for you ====================
#  After training the model, you can now make recommendations by computing
#  the predictions matrix.

p = np.dot(X, Theta.T)
my_predictions = p[:, 0] + Ymean

movieList = loadMovieList()

ix = np.argsort(my_predictions)[::-1]
print('\nTop recommendations for you:')
for i in range(10):
    j = ix[i]
    print('Predicting rating {:.1f} for movie {:s}'.format(
        my_predictions[j], movieList[j]))

print('\n\nOriginal ratings provided:')
for i, my_rating in enumerate(my_ratings):
    if my_rating > 0:
        print('Rated {:d} for {:s}'.format(int(my_rating), movieList[i]))
