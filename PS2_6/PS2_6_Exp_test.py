import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from numpy import linalg
from numpy import matrix
import math
import numpy.matlib

testing_data = pd.read_csv('boosting-test.csv', header = None)
Y = testing_data[0]
print "starting from here!!!"
	
T = 10
alpha = [ 0.42648979, 0.40682636, 0.33340693, 0.34141874, 0.30481865, 0.30864987, 0.3006796, 0.30501678, 0.26806364]
features = [0, 6, 9, 17, 15, 9, 14, 12, 1]
medians = [0.791664360502, 0.773228226793, -0.0804007341546, 0.168715010049, 0.878823246817, -0.0804007341546, 0.9107438107, -0.0804007341546, 0.837679334033]
countRight =0
countWrong =0
H = np.zeros(len(testing_data))
for index in range(0, len(testing_data)):
	if testing_data[1][index] <= medians[0]:
		H[index] += (-1) * alpha[0]
	else:
		H[index] += alpha[0]
	if testing_data[7][index] <= medians[1]:
		H[index] += (-1) * alpha[1]
	else:
		H[index] += alpha[1]	
	if testing_data[10][index] <= medians[2]:
		H[index] += (-1) * alpha[2]
	else:
		H[index] += alpha[2]	
	if testing_data[18][index] <= medians[3]:
		H[index] += (-1) * alpha[3]
	else:
		H[index] += alpha[3]	
	if testing_data[16][index] <= medians[4]:
		H[index] += (-1) * alpha[4]
	else:
		H[index] += alpha[4]	
	if testing_data[10][index] <= medians[5]:
		H[index] += (-1) * alpha[5]
	else:
		H[index] += alpha[5]	
	if testing_data[15][index] <= medians[6]:
		H[index] += (-1) * alpha[6]
	else:
		H[index] += alpha[6]	
	if testing_data[13][index] <= medians[7]:
		H[index] += (-1) * alpha[7]
	else:
		H[index] += alpha[7]	
	if testing_data[2][index] <= medians[8]:
		H[index] += (-1) * alpha[8]
	else:
		H[index] += alpha[8]
	if H[index] < 0 and Y[index] < 0:
		countRight += 1
	elif H[index] > 0 and Y[index] > 0:
		countRight += 1
	else:
		countWrong += 1

print "% right prediction = ", float(countRight)/(countRight + countWrong)
print "no of rights = ", countRight
print "no of wrongs = ", countWrong