from utilFuncs import invert
import math
def read_files_for_X(file, X):
	line = (file.readline()).strip()
	while line:
		x0 = 1
		x1 = float(line[0:line.index(' ')])
		x2 = None
		offset = None
		if line[line.index(' ') + 2] == ' ':
			offset = 3
		else:
			offset = 2
		x2 =  float(line[line.index(' ') + offset:len(line)])
		line = (file.readline()).strip()
		x = [x0, x1, x2]
		X.append(x)
	#print X
def read_files_for_Y(file, Y):
	line = (file.readline()).strip()
	while line:
		y = [float(line)]
		line = (file.readline()).strip()
		Y.append(y)	

def multiply(matrix1, matrix2, res):
	for i in range(len(matrix1)):
		for j in range(len(matrix2[0])):
			for k in range(len(matrix2)):
				res[i][j] += matrix1[i][k] * matrix2[k][j]

				
fileX = open('logistic_x.txt', 'r')
X = []
read_files_for_X(fileX, X)
Xt = [[0 for i in xrange(99)] for i in xrange(3)]
for i in range(len(X)):
	for j in range(len(X[0])):
		Xt[j][i] = X[i][j]
res1 = [[0 for i in xrange(3)] for i in xrange(3)]

multiply(Xt, X, res1)
#print res1
res2 = invert(res1)
res3 = [[0 for i in xrange(99)] for i in xrange(3)]
multiply(res2, Xt, res3)

fileY = open('logistic_y.txt', 'r')
Y = []
read_files_for_Y(fileY, Y)
W = [[0 for i in xrange(1)] for i in xrange(3)]
multiply(res3, Y, W)
#print W[0][0], W[1][0], W[2][0]
print W
H = []
for i in range(len(X)):
	h = 0
	for j in range(len(W)):
		h += W[j][0] * X[i][j]
	print h
	if h < 0:
		H.append(-1)
	else:
		H.append(1)
detW = math.sqrt((W[1][0] * W[1][0]) +  (W[2][0] * W[2][0]))
W[0][0] = W[0][0] / detW
W[1][0] = W[1][0] / detW
W[2][0] = W[2][0] / detW

print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
res4 = [[0 for i in xrange(1)] for i in xrange(99)]
minGeoMargin = 100
minI = -1
ctr = 0
for i in range(len(X)):
	res4[i][0] = W[0][0] + W[1][0] * X[i][1] + W[2][0] * X[i][2]
	print res4[i][0], Y[i][0], res4[i][0] * Y[i][0]
	if res4[i][0] * Y[i][0] < 0:
		ctr += 1
	if minGeoMargin > res4[i][0] * Y[i][0] and (res4[i][0] * Y[i][0] > 0):
		minGeoMargin = res4[i][0] * Y[i][0]
		minI = i

print "min geometric margin = ",  minGeoMargin, " value of i = ", X[minI], " no of outliers = ", ctr
