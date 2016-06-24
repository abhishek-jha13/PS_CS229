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
s = pd.Series(0.0, index = df.index.values.tolist())
c = 0.01
theta = matrix( [[0.12519154306],[4860 *c],[c]] )
for index in df.index.values.tolist():
	#s[index] = 1/(theta[0, 0] + theta[1, 0] * float(index) + theta[2, 0] * float(index) * float(index))
	s[index] = 1/(theta[0, 0] + theta[1, 0] * (float(index) - 1215.0) + theta[2, 0] * (float(index) -1215.0) * (float(index) - 1215.0))
	print index, " : ", s[index], "  ", df[index]
#df.plot()
s.plot()
plt.show()
