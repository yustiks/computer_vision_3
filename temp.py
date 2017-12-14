import numpy as np

a = np.array([1,2,3,4,5,1,2,3,1,2,1])
b = np.histogram(a,range(10))
print (b)