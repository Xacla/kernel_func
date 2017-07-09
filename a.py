import numpy as np
a=np.array([[4,2,3],[2,3,1]])
b=np.array([1,1,1])
b=b[:,np.newaxis]
print(a.shape)
print(b.shape)
print(np.dot(a,b))
