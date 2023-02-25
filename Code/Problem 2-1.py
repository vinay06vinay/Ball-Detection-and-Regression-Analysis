import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import os 

a= np.loadtxt("pc1.csv", delimiter=",", dtype = float)
x= a[:,0]
y = a[:,1]
z= a[:,2]
x_mean = np.mean(x)
y_mean = np.mean(y)
z_mean = np.mean(z)
'''
Problem 2-1-a - Computing the Covariance Matrix
'''

#Covariance of X_X
cov_x_x = sum((x-x_mean)*(np.transpose(x-x_mean))) / x.shape[0]
print(f"Covariance of cov(x,x) {cov_x_x}")
#Covariance of Y_Y
cov_y_y = sum((y-y_mean)*(np.transpose(y-y_mean))) / y.shape[0]
print(f"Covariance of cov(y,y) {cov_y_y}")
#Covariance of Z_Z
cov_z_z = sum((z-z_mean)*(np.transpose(z-z_mean))) / z.shape[0]
print(f"Covariance of cov(z,z) {cov_z_z}")
#Covariance of x_y
cov_x_y = sum((x-x_mean)*(y-y_mean)) / x.shape[0]
print(f"Covariance of cov(x,y) {cov_x_y}")
#Covariance of x_z
cov_x_z = sum((x-x_mean)*(z-z_mean)) / x.shape[0]
print(f"Covariance of cov(x,z) {cov_x_z}")
#Covariance of y_z
cov_y_z = sum((y-y_mean)*(z-z_mean)) / x.shape[0]
print(f"Covariance of cov(y,z) {cov_y_z}")
cov_xyz = np.array([[cov_x_x,cov_x_y,cov_x_z],[cov_x_y,cov_y_y,cov_y_z],[cov_x_z,cov_y_z,cov_z_z]])
print(f"The Covariance Matrix for the three point cloud of (x,y,z):\n{cov_xyz}")

'''
Problem 2-1-b - Computing Surface Normal with covariance matrix derived

1. First eigen values and eigen vectors are of the covariance matrix are calculated
2. The index of minimum eigen value is stored
3. The eigen vector corresponding to minimum eigen value is the direction of surface normal and magnitude is derived from norm of eigen 
vector corresponding to minimum eigen value
'''
eig_values,eig_vectors = np.linalg.eig(cov_xyz)
eig_values_min_index = np.argmin(eig_values)
eig_vector_s = eig_vectors[:,eig_values_min_index]
surface_normal_magnitude = np.sqrt(np.sum(eig_vector_s**2))
print("The Direction of surface normal is in the direction of eigen vector : ",eig_vector_s)
print("The magnitude of surface normal is:",surface_normal_magnitude)
