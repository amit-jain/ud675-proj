"""
Runs iterations over the Decision Trees model with the chosen complexity.
"""

from numpy import *
import pylab as pl
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor

# Load the boston dataset
boston = datasets.load_boston()

# We will vary the depth of decision trees from 2 to 25
max_depth = arange(2, 25)
train_err = zeros(400)
test_err = zeros(400)

preds = []

# Run 400 iterations
for i in range(400):
	# Seperate the data into training and testing set
	X, y = shuffle(boston.data, boston.target)
	offset = int(0.7*len(X))
	X_train, y_train = X[:offset], y[:offset]
	X_test, y_test = X[offset:], y[offset:]
	
	# Setup a Decision Tree Regressor so that it learns a tree with depth d
	regressor = DecisionTreeRegressor(max_depth=25)
	regressor.fit(X_train, y_train)

	train_err[i] = mean_squared_error(y_train, regressor.predict(X_train))
	test_err[i] = mean_squared_error(y_test, regressor.predict(X_test))
	
	# Analyze the data on this model
	data = [[11.95, 0.00, 18.100, 0, 0.6590, 5.6090, 90.00, 1.385, 24, 680.0, 20.20, 332.09, 12.13]]
	datay = regressor.predict(data)
	preds.append(datay)
	
	print "Prediction for " + str(data) + " = " + str(datay)
	print " test error = ", test_err[i], " train error = ", train_err[i]    

print "Variance for prediction ", var(preds)
print "Max value for prediction ", max(preds)
print "Min value for prediction ", min(preds)
print "Mean value for predictions ", mean(preds)
print "Mean value for test error ", mean(test_err)
print "Mean value for train error ", mean(train_err)