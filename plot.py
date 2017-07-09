import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#読み込み
example=pd.read_csv(r"C:\Users\owner\学校関連\sotuken\sample.csv")
example=example.as_matrix()
x_dimension=2 #次元の数
beta=1
y=np.array(example[:,1])
#y=np.matrix(y)

#サンプルに対するn次元Xベクトルの生成
vector_x=[]
for i in range(example.shape[0]):
    a=[]
    for k in range(x_dimension):
        a.append(example[i][0]**(k+1))
    vector_x.append(a)
vector_x=np.array(vector_x)
print(vector_x)

#カーネル行列の生成
kernel_matrix=[]
for i in range(example.shape[0]):
    k_peace=[]
    for j in range(example.shape[0]):
        #ガウスカーネルの生成
        y_i=vector_x[j][:]-vector_x[i][:]
        y_i=np.linalg.norm(y_i)
        y_i=np.exp(-beta*y_i**2)
        k_peace.append(y_i)
    kernel_matrix.append(k_peace)
kernel_matrix=np.array(kernel_matrix)
#print(kernel_matrix)

#alpha行列を求める
kernel_matrix_r=np.linalg.inv(kernel_matrix)
print(kernel_matrix_r.shape)
print(y.shape)
alpha=np.dot(kernel_matrix_r,y)
print(alpha)

#求めた行列から値を求める
x_range=100
x_min=-2
x_max=2
hoge=[i/x_range+x_min for i in range((x_max-x_min)*x_range)]
print(hoge)

plt.scatter(example[:,0],example[:,1])
plt.show()
