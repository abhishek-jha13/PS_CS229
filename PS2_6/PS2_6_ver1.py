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
print "starting from here!!!"
def getCountOfErrorForAllFeatures(D, i, training_data, Y):
	global low_correct
	global low_incorrect
	global high_correct
	global high_incorrect
	
	low_correct = np.zeros(len(training_data.columns[1:]))
	low_incorrect = np.zeros(len(training_data.columns[1:]))
	high_correct = np.zeros(len(training_data.columns[1:]))
	high_incorrect = np.zeros(len(training_data.columns[1:]))

	minErrorFeature = 1
	minError = 1
	for index1 in training_data.columns[1:]:
		X = training_data[index1]
		median_X = X.median()	
		for index2 in training_data.index.values.tolist():
			if X[index2] <= median_X and Y[index2] == 1.0:
				low_incorrect[index1 - 1] += D[i, index2]
			elif X[index2] <= median_X and Y[index2] == -1.0:
				low_correct[index1 - 1] += D[i, index2]
			elif X[index2] > median_X and Y[index2] == 1.0:
				high_correct[index1 - 1] += D[i, index2]
			elif X[index2] > median_X and Y[index2] == -1.0:
				high_incorrect[index1 - 1] += D[i, index2]
		sum = low_correct[index1 - 1] + low_incorrect[index1 - 1] + high_correct[index1 - 1] + high_incorrect[index1 - 1]
	print "insdie line 36"
	print "low_incorrect = ", low_incorrect
	print "low_correct = ", low_correct
	print "high_incorrect = ", high_incorrect
	print "high_correct = ", high_correct
#		X_error = (low_incorrect + high_incorrect)/float(len(training_data))
#		print X_error
#	return minErrorFeature
def getMinErrorFeature(low_correct, low_incorrect, high_correct, high_incorrect):
	minIndex = 0
	min = low_incorrect[0] + high_incorrect[0]
	for index in range(0, len(low_incorrect)):
		if (low_incorrect[index] + high_incorrect[index]) < min:
			minIndex = index
			min = low_incorrect[index] + high_incorrect[index]
	return minIndex
#def setIndixesToZeroAgain()
	
low_correct = np.zeros(len(training_data.columns[1:]))
low_incorrect = np.zeros(len(training_data.columns[1:]))
high_correct = np.zeros(len(training_data.columns[1:]))
high_incorrect = np.zeros(len(training_data.columns[1:]))
T = 10
D = np.zeros([T, len(training_data)])
error = np.zeros([T])
alpha = np.zeros([T])

for j in range(0, len(D[0])):
	D[0, j] = 1.0/len(D[0])

for i in range(1, len(D)):
	print "inside adaboost start"
	getCountOfErrorForAllFeatures(D, i - 1, training_data, Y)
	print "inside line 63"
	#print low_incorrect
	print high_incorrect + low_incorrect
	minErrorFeatureIndex = getMinErrorFeature(low_correct, low_incorrect, high_correct, high_incorrect)
#	setIndixesToZeroAgain()
	print "min error feature = ", minErrorFeatureIndex
	print "sssss = ", low_incorrect[minErrorFeatureIndex], high_incorrect[minErrorFeatureIndex]
	error[i - 1] = float(low_incorrect[minErrorFeatureIndex] + high_incorrect[minErrorFeatureIndex])
	print "error = ", error[i - 1]
	alpha[i - 1] = 0.5 * math.log((1 - error[i - 1]) / error[1 - 1])
	minError_X = training_data[minErrorFeatureIndex + 1]
	sum = 0
	d1 =0
	d2 =0
	d3 =0
	d4 =0
	ctrd1 =0
	ctrd2 =0
	ctrd3 =0
	ctrd4 =0
	for j in range(0, len(D[0])):
		#print "minError_X[", j, "] = ", minError_X[j]
		if (minError_X[j] <= minError_X.median()) and Y[j] == 1.0 :
			D[i, j] = 0.5 * D[i - 1, j]/error[i - 1]
			ctrd1 += 1
			d1 += 0.5 * D[i - 1, j]/error[i - 1]
		elif (minError_X[j] <= minError_X.median()) and Y[j] == -1.0:
			D[i, j] = 0.5 * D[i - 1, j]/(1 - error[i - 1])
			ctrd2 += 1
			d2 += 0.5 * D[i - 1, j]/(1 - error[i - 1])
		elif (minError_X[j] > minError_X.median()) and Y[j] == -1.0:
			D[i, j] = 0.5 * D[i - 1, j]/error[i - 1]
			ctrd3 += 1
			d3 += 0.5 * D[i - 1, j]/error[i - 1]
		elif (minError_X[j] > minError_X.median()) and Y[j] == 1.0:
			D[i, j] = 0.5 * D[i - 1, j]/(1 - error[i - 1])
			ctrd4 += 1
			d4 += 0.5 * D[i - 1, j]/(1 - error[i - 1])
		sum += D[i, j]
	print "sum of all D[", i, "] = ", sum
	print "sum of all d1[", i, "] = ", d1
	print "sum of all d2[", i, "] = ", d2
	print "sum of all d3[", i, "] = ", d3
	print "sum of all d4[", i, "] = ", d4
	print "sum of all ctrd1[", i, "] = ", ctrd1
	print "sum of all ctrd2[", i, "] = ", ctrd2
	print "sum of all ctrd3[", i, "] = ", ctrd3
	print "sum of all ctrd4[", i, "] = ", ctrd4
print "error : " , error
print "alpha : ", alpha			
