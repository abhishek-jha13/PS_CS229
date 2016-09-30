import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from numpy import linalg
from numpy import matrix
import math
import numpy.matlib

training_data = pd.read_csv('boosting-train.csv', header = None)
Y = training_data[0]
medians = np.zeros(len(training_data))
features = [0, 6, 9, 17, 15, 9, 14, 12, 1]

medians[0] = training_data[1].median()
medians[1] = training_data[7].median()
medians[2] = training_data[10].median()
medians[3] = training_data[18].median()
medians[4] = training_data[12].median()
medians[5] = training_data[10].median()
medians[6] = training_data[16].median()
medians[7] = training_data[10].median()
medians[8] = training_data[15].median()

print medians[0]
print medians[1]
print medians[2]
print medians[3]
print medians[4]
print medians[5]
print medians[6]
print medians[7]
print medians[8]

print "_______________________________________________________________--"
for i in range(1, len(training_data.columns[1:]) + 1):
	print training_data[i].median()
