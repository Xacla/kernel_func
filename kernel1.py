import numpy as np
import matplotlib.pyplot as plt
x=3
range_x=100
x_2=[i/range_x for i in range(x*2*range_x)]
beta=1
kernel=[0 for i in range(x*2*range_x)]

for i in range(x*2*range_x):
    z= np.linalg.norm(x_2[i] - x)
    kernel[i]=np.exp(-beta*z**2)

plt.plot(x_2,kernel)
#plt.show()
filename='example1.png'
plt.savefig(filename)
