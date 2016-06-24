import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from numpy import linalg
from numpy import matrix
import math
from utilFuncs import invert
import numpy.matlib

def gaussian_kernel(x0, x, c, a=1.0):
    """
    Gaussian kernel.

    :Parameters:
      - `x`: nearby datapoint we are looking at.
      - `x0`: data point we are trying to estimate.
      - `c`, `a`: kernel parameters.
    """
    # Euclidian distance
    diff = x - x0
    dot_product = diff * diff.T
    return a * np.exp(dot_product / (-2.0 * c**2))


def get_weights(training_inputs, datapoint, c):
    x = np.mat(training_inputs)
    n_rows = x.shape[0]
    # Create diagonal weight matrix from identity matrix
    weights = np.mat(np.eye(n_rows))
    for i in xrange(n_rows):
        weights[i, i] = gaussian_kernel(datapoint, x[i], c)
        #print weights[i, i]

    return weights


training_data = pd.read_csv('quasar_train.csv')
training_data  = training_data.mean()
x = np.matlib.zeros([len(training_data), 1])
y = np.matlib.zeros([len(training_data), 1])
ctr  = 0
datapoint = 1599.000

for index in training_data.index.values.tolist():
	x[ctr, 0] = float(index)
	y[ctr, 0] = training_data[index]
	ctr += 1
c = 1
y_estimated = pd.Series(0.0, index = training_data.index.values.tolist())

test_data = pd.read_csv('quasar_test.csv')
test_data  = test_data.mean()

for index in training_data.index.values.tolist():
	datapoint = float(index)
	weights = get_weights(x, datapoint, c)
	xt = (x.T * weights) * x
	betas = ((xt.I * x.T) * weights) * y
	y_estimated[index] = datapoint * betas
	print index, " : ", y_estimated[index], " : ", test_data[index[0:len(index) - 1]]

training_data.plot()
y_estimated.plot()
test_data.plot()
plt.show()
	
