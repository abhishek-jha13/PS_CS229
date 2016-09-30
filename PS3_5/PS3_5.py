from scipy import misc
import matplotlib.pyplot as plt
import random
import numpy as np

# image = misc.face()
image = misc.imread('sidhu.jpg')
plt.imshow(image)
plt.show()
centroids = np.zeros((16, 3), dtype=int)
C = np.zeros((len(image), len(image[0])), dtype=int)

for i in range(0, len(centroids)):
    centroids[i] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
print (centroids)
# C = numpy.zeros((len(centroids)))
ctrOfClusters = np.zeros((len(centroids)), dtype=int)
sumOfClusters = np.zeros((len(centroids), 3))
ctr = 10
while (ctr != 0):
    ctr -= 1
    for i in range(0, len(image)):
        for j in range(0, len(image[0])):
            C[i, j] = 0
            minDist = np.linalg.norm(image[i, j] - centroids[C[i][j]])
            for k in range(1, len(centroids)):
                if np.linalg.norm(image[i, j] - centroids[k]) < minDist:
                    C[i, j] = k
                    minDist = np.linalg.norm(image[i, j] - centroids[C[i, j]])
            sumOfClusters[C[i, j]] += image[i, j]
            ctrOfClusters[C[i, j]] += 1
    for i in range(0, len(centroids)):
        if ctrOfClusters[i] != 0:
            centroids[i] = sumOfClusters[i] / ctrOfClusters[i]
        else:
            centroids[i] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
    print ("######################################")
    print (centroids)
for i in range(0, len(image)):
    for j in range(0, len(image[0])):
        image[i, j] = centroids[C[i, j]]
plt.imshow(image)
plt.show()
