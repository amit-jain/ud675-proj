"""
Runs iterations on a Neural Network model with the chosen complexity (hidden layers).
We use pybrain (http://pybrain.org/) to design and train our NN
"""

from numpy import *
import pylab as pl
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from pybrain.structure import FeedForwardNetwork
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

# Load the boston dataset
boston = datasets.load_boston()
train_err = zeros(25)
test_err = zeros(25)
preds = []
	
# Run 25 iterations
for i in range(25):
	# Seperate the data into training and testing set	
	X, y = shuffle(boston.data, boston.target)
	offset = int(0.7*len(X))
	X_train, y_train = X[:offset], y[:offset]
	X_test, y_test = X[offset:], y[offset:]
	
	# List all the different networks we want to test again
	# All networks have 13 input nodes and 1 output nodes
	# All networks are fully connected
	net = []
	# 1 hidden layer with 1 node
	net.append(buildNetwork(13,1,1))
	# 1 hidden layer with 5 nodes
	net.append(buildNetwork(13,5,1))
	# 2 hidden layers with 7 and 3 nodes resp
	net.append(buildNetwork(13,7,3,1))
	# 3 hidden layers with 9, 7 and 3 nodes resp
	net.append(buildNetwork(13,9,7,3,1))
	# 4 hidden layers with 9, 7, 3 and 2 noes resp
	net.append(buildNetwork(13,9,7,3,2,1))
	net_arr = range(0, len(net))
	
	# The dataset will have 13 features and 1 target label
	ds = SupervisedDataSet(13, 1)
		
	# We will train each NN for 50 epochs
	max_epochs = 50
	
	# Convert the boston dataset into SupervisedDataset
	for j in range(1, len(X_train)):
		ds.addSample(X_train[j], y_train[j])
	
	trainer = BackpropTrainer(net[4], ds)

	# Run backprop for max_epochs number of times
	for k in range(1, max_epochs):
		train_err[i] = trainer.train()

	# Find the labels for test set
	y = zeros(len(X_test))

	for j in range(0, len(X_test)):
		y[j] = net[4].activate(X_test[j])

    # Calculate MSE for all samples in the test set
	test_err[i] = mean_squared_error(y, y_test)

	# Analyze the data on this model
	data = [11.95, 0.00, 18.100, 0, 0.6590, 5.6090, 90.00, 1.385, 24, 680.0, 20.20, 332.09, 12.13]
	datay = net[4].activate(data)
	preds.append(datay)
	
	print "Prediction for " + str(data) + " = " + str(datay)
	print " test error = ", test_err[i], " train error = ", train_err[i]
	
print "Variance for prediction ", var(preds)
print "Max value for prediction ", max(preds)
print "Min value for prediction ", min(preds)
print "Mean value for predictions ", mean(preds)
print "Mean value for test error ", mean(test_err)
print "Mean value for train error ", mean(train_err)
