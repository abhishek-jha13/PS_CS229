import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from numpy import linalg
from numpy import matrix
import math
from utilFuncs import invert
import numpy.matlib
df = pd.read_csv('quasar_train.csv')
df  = df.mean()
#indexListOfX = df.index.values.tolist()
c = 1
theta = matrix( [[0.12519154306],[4860 *c],[c]] )
#theta = matrix( [[1],[1],[0.0009]] )
ctr = 0
while (ctr < 10):
	
	ctr += 1
	dL = np.matlib.zeros([3, 1])
	for index in df.index.values.tolist():
		inputX = float(index)
		T1 = (1 / (theta[0] + theta[1] * (inputX - 1215) + theta[2] * (inputX -1215) * (inputX - 1215))) - df[index]
		T2 = 1 / math.pow(1 + 1 * (inputX - 1215) + 1 * (inputX - 1215) * (inputX - 1215), 2)
		dL[0, 0] += (-1) * T1 * T2 
		dL[1, 0] += (-1) * T1 * T2 * inputX
		dL[2, 0] += (-1) * T1 * T2 * inputX * inputX
	H = np.matlib.zeros([3, 3])
	for index in df.index.values.tolist():
		inputX = float(index)
		T1 = 3 / math.pow(theta[0] + theta[1] * (inputX - 1215) + theta[2] * (inputX - 1215) * (inputX - 1215), 4)
		T2 = 2 * df[index] / math.pow(theta[0] + theta[1] * (inputX - 1215) + theta[2] * (inputX - 1215) * (inputX - 1215), 3)
		H[0, 0] += (T1 - T2)
		H[1, 0] += (T1 - T2) * (inputX - 1215)
		H[2, 0] += (T1 - T2) * (inputX - 1215) * (inputX - 1215)
		
		H[0, 1] += (T1 - T2) * (inputX - 1215)
		H[1, 1] += (T1 - T2) * (inputX - 1215) * (inputX - 1215)
		H[2, 1] += (T1 - T2) * (inputX - 1215) * (inputX - 1215) * (inputX - 1215)

		H[0, 2] += (T1 - T2) * (inputX - 1215) * (inputX - 1215)
		H[1, 2] += (T1 - T2) * (inputX - 1215) * (inputX - 1215) * (inputX - 1215)
		H[2, 2] += (T1 - T2) * (inputX - 1215) * (inputX - 1215) * (inputX - 1215) * (inputX - 1215)
	#print H
	#H_inverse = np.linalg.inv(H)
	H_inverse = H.I
	#print H_inverse
	#print dL
	theta = theta - H_inverse * dL
	print theta
	#df.plot()
	#plt.show()
	a = 1/(theta[0, 0] + theta[1, 0] * 1150 + theta[2, 0] * 1150 * 1150)
	print a
s = pd.Series(0.0, index = df.index.values.tolist())
for index in s.index.values.tolist():
	#s[index] = 1/(theta[0, 0] + theta[1, 0] * float(index) + theta[2, 0] * float(index) * float(index))
	s[index] = 1/(0.12519154306 - 0.0277529 * (float(index) - 1215.0) + 0.0000846698 * (float(index) -1215.0) * (float(index) - 1215.0))
#print s
df.plot()
s.plot()
plt.show()
	
