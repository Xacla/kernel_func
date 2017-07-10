import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def create_vector(x,x_dimension):
    vector=[]
    for i in range(x.shape[0]):
        a=[]
        for k in range(x_dimension):
            a.append(x[i]**(k+1))
        vector.append(a)
    vector=np.array(vector)
    #print(vector)
    return vector

#読み込み
#example=pd.read_csv(r"C:\Users\owner\学校関連\sotuken\sample.csv")
example=pd.read_csv("/Users/tomita/Desktop/研究/カーネル主成分分析/sample.csv")
example=example.as_matrix()
x_dimension=2 #次元の数
beta=1
x=np.array(example[:,0])
y=np.array(example[:,1])
lamda=0.001
#y=np.matrix(y)

#サンプルに対するn次元Xベクトルの生成
vector_x=create_vector(x,x_dimension)

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
I_n=np.eye(example.shape[0])
I_n=lamda*I_n
kernel_matrix_plus=kernel_matrix+I_n #λを入れて計算する場合
#kernel_matrix_r=np.linalg.inv(kernel_matrix)　#λを入れないで計算しない場合
kernel_matrix_r=np.linalg.inv(kernel_matrix_plus)
print(kernel_matrix_r.shape)
print(y.shape)
alpha=np.dot(kernel_matrix_r,y)
print(alpha)

#求めた行列から値を求める
x_range=100
x_min=-2
x_max=2
hoge=[i/x_range+x_min for i in range((x_max-x_min)*x_range)]
hoge.append(2.0)
hoge=np.array(hoge)
vector_hoge=create_vector(hoge,x_dimension)

hoge_y=[]
for i in range(hoge.shape[0]):
    sum_all=0
    for j in range(x.shape[0]):
        y_i=vector_hoge[i][:]-vector_x[j][:]
        y_i=np.linalg.norm(y_i)
        y_i=np.exp(-beta*y_i**2)
        sum_all=sum_all+alpha[j]*y_i
    hoge_y.append(sum_all)


plt.scatter(example[:,0],example[:,1])
plt.plot(hoge,hoge_y)
plt.show()
