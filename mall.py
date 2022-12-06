import numpy as np 
from sklearn.datasets import make_blobs  
import matplotlib.pyplot as plt
import random
import math
import collections as cl 
import pandas as pd 
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)                 

df = pd.read_csv(r'Mall_Customers.csv')
print(df)
data = []
# data.append(np.array(df["Age"][:]))
data.append(np.array(df["Annual Income (k$)"][:]))

data.append(np.array(df["Spending Score (1-100)"][:]))
dataset = np.array(data).transpose()



K = 10
# K_prime = 50
n_samples = 1000
index_dispersion = 1
err_list = []

"""by this function you can 
generate centers of your datasets"""
def center(K):
  nums=[]
  nums2=[]
  for i in range(K):
    nums2=[0,0]
    for j in range(2):
      nums2[j]=round(random.random(),2)*index_dispersion
    nums.append(nums2)
  return np.array(nums)





"""This function sets the dispersion of our clusters"""
def center_std(K):
  center_std=[]
  for i in range(K):
    center_std.append(round(random.random(),2))
  return center_std



"""In this function, we take default centers and data as input and

check which center each data is closest to.

Then we attribute that data to the nearest center

then we can return the clustered_Data as output of the function"""

def k_mean(Data_in,center):
    # print(Data_in[2])
    clustered_Data=[[] for i in range(K_prime)]
    for i in range(len(Data_in)):
        distances=[]
        for j in range(len(center)):
            distances.append(math.dist(Data_in[i],center[j]))
        clustered_Data[distances.index(min(distances))].append(Data_in[i].tolist())
    return clustered_Data
  
  
  

def center_update(clustered_Data,center):
    print(len(clustered_Data))
    for i in range(len(clustered_Data)):
        nparr = np.array(clustered_Data[i])
        print(center,"\n****************")
        print(nparr.all())
        if len(nparr) != 0:
            center[i]=[np.average(nparr[:,0]),np.average(nparr[:,1])]
            
    return center




def center_init(X,y,K):
  init_points = [0 for i in range(K)]
  for j in range(K):
    for i in range(len(y)):
      if(y[i] == j):
        init_points[j] = X[i].tolist()
        break
  # print(init_points)
  return np.array(init_points)
    
    
    
    
blob_center = center(K)
cluster_std = center_std(K)
# X, y = make_blobs(n_samples = n_samples, centers=blob_center,random_state=10,cluster_std= cluster_std)
X  = dataset


for K_prime in range(1,16):
    center_k_prime = center(K_prime)
    # center_starter = center_init(X,y,K)
    """scattering inital points for cluster"""
    # plt.scatter(X[:,0] , X[:,1])
    # plt.scatter(center_k_prime[:,0] , center_k_prime[:,1])
    # plt.show()
    clustered_Data=k_mean(X,center_k_prime)
    center_old=np.array([[1,1] for i in range(K_prime)])
    while True :
        if (center_old==center_k_prime).all():
            break
        center_buffer=center_update(clustered_Data,center_k_prime)
        center_old=center_k_prime
        center_k_prime=center_buffer
        clustered_Data=k_mean(X,center_k_prime)
        # plt.scatter(X[:,0] , X[:,1])
        # plt.scatter(center_k_prime[:,0] , center_k_prime[:,1],color="red")
        # plt.show()
    """"ploting process"""
    
    """uncomment the following lines if you want to see the steps of the test """
    for i in range(K_prime):
        twodarr = np.array(clustered_Data[i])
        rand_colors = ["#"+''.join([random.choice('ABCDEF0123456789') for i in range(6)])]
        if len(twodarr) != 0:
            plt.scatter(twodarr[:,0],twodarr[:,1],color=rand_colors )
    plt.show()
    sum_up = 0 
    for i in range(len(clustered_Data)):
        for j in range(len(clustered_Data[i])):
            sum_up = sum_up + math.dist(clustered_Data[i][j],center_k_prime[i])
    err = sum_up/((i+1)*(j+1))
    err_list.append(err)



plt.plot(err_list)
plt.title('Error Count')
plt.show()


# print(clustered_Data.shape)
# print(arr[0, :, 0])
# plt.scatter(arr , arr)
# plt.show()






