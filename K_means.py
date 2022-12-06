import numpy as np 
from sklearn.datasets import make_blobs  
import matplotlib.pyplot as plt
import random
import math
import collections as cl 
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)                 


"""for testing process we shoud change k to 2,3,4 and run the code"""
K = 4
n_samples = 10000
index_dispersion = 10
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
    clustered_Data=[[] for i in range(K)]
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
X, y = make_blobs(n_samples = n_samples, centers=blob_center,random_state=10,cluster_std= cluster_std)
center_starter = center_init(X,y,K)
"""scattering inital points for cluster"""
plt.scatter(X[:,0] , X[:,1])
plt.scatter(center_starter[:,0] , center_starter[:,1])
plt.show()
clustered_Data=k_mean(X,center_starter)
center_old=np.array([[1,1] for i in range(K)])
while True :
  if (center_old==center_starter).all():
    break
  center_buffer=center_update(clustered_Data,center_starter)
  center_old=center_starter
  center_starter=center_buffer
  clustered_Data=k_mean(X,center_starter)
  plt.scatter(X[:,0] , X[:,1])
  plt.scatter(center_starter[:,0] , center_starter[:,1],color="red")
  plt.show()
""""ploting process"""
sum_up = 0 
for i in range(K):
    twodarr = np.array(clustered_Data[i])
    rand_colors = ["#"+''.join([random.choice('ABCDEF0123456789') for i in range(6)])]
    plt.scatter(twodarr[:,0],twodarr[:,1],color=rand_colors )
plt.show()
# for i in range(len(clustered_Data)):
#   for j in range(len(clustered_Data[i])):
#     sum_up = sum_up + math.dist(clustered_Data[i][j],center_starter[i])
# err = sum_up/(i*j)
# err_list.append(err)



# plt.plot(err_list)
# plt.title('Error Count')
# plt.show()


# print(clustered_Data.shape)
# print(arr[0, :, 0])
# plt.scatter(arr , arr)
# plt.show()






