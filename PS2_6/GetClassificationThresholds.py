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

X = training_data[1].copy(deep=True)
X = X.sort_values(inplace=True)
sumFromLeft = X.sum()
sumFromRight = 0
countRight = 0
countWrong = 0
for i in range(0, len(X)):
	if X[i] == 1:
		