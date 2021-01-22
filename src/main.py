"""
Greatly inspired by (and beautifully explained in):
https://cs231n.github.io/neural-networks-2/
"""

from methods import * 
import matplotlib.pyplot as plt

ds = 128
ne = 2
data=np.random.normal(loc=6, scale=1, size=(ds,ne))
data[...,1] = 2*data[...,0] + data[...,1]

kernel=sangers_pca(data.copy())
U,S,V = svd(data.copy())

print("Sangers kernel: \n", np.around(kernel, decimals=2))
print("svd_U: \n", np.around(U, decimals=2))
print("svd_S: \n", np.around(S, decimals=2))
print("svd_V: \n", np.around(V, decimals=2))
print("svd_V/S \n", np.around(U/S, decimals=2))


data_cpy = data.copy()

fig, ax = plt.subplots(2)
ax[0].scatter(data[...,0],data[...,1],c='#ff0000',label='Original')
data -= np.mean(data,axis=0)
ax[0].scatter(data[...,0],data[...,1],c='#00ff00',label='Mean-shifted')
data = np.dot(data,U)
ax[0].scatter(data[...,0],data[...,1],c='#0000ff',label='SVD decorrelation')
data1 = data.copy() / np.std(data,axis=0)
ax[0].scatter(data1[...,0],data1[...,1],c='#ffff00',label='STD scaled')
data /= S
print(np.std(data,axis=0))
ax[0].scatter(data[...,0],data[...,1],c='#880000',label='Whitened')
ax[0].set_title("SVD")
ax[0].legend()


data = data_cpy
ax[1].scatter(data[...,0],data[...,1],c='#ff0000',label='Original')
data -= np.mean(data,axis=0)
ax[1].scatter(data[...,0],data[...,1],c='#00ff00',label='Mean-shifted')
data = np.dot(data,kernel)
ax[1].scatter(data[...,0],data[...,1],c='#0000ff',label='Sangers decorrelation')
#data *= np.std(data,axis=0)**2
#ax[1].scatter(data[...,0],data[...,1],c='#880000',label='Original')
ax[1].set_title("Sangers Kernel")
ax[1].legend()


ax[0].grid('on')
ax[1].grid('on')
plt.show()