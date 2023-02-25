

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def plot_standard_least_square(x,y,z,z_final,file):
    fig, ax = plt.subplots()
    ax = plt.axes(projection='3d')
    ax.scatter3D(x, y, z, color='m',s=5)
    ax.plot(x, y, z_final, '--b')
    plt.title("Standard Least Square Curve for " + file)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    plt.show()
def compute_standard_least_square(file):
    '''
    Standard Least squares is computed by taking the a plane ax+by+c = z
    Equations and matrix derived in the report
    '''
    a= np.loadtxt(file, delimiter=",", dtype = float)
    x= a[:,0]
    y = a[:,1]
    z= a[:,2]
    x_sum = np.sum(x)
    y_sum = np.sum(y)
    z_sum = np.sum(z)
    x_2_sum = np.sum(np.power(x,2))
    y_2_sum = np.sum(np.power(x,2))
    x_y_sum = np.sum(x*y)
    x_z_sum = np.sum(x*z)
    y_z_sum = np.sum(y*z)
    a_matrix = np.array([[x_2_sum,x_y_sum,x_sum],[x_y_sum,y_2_sum,y_sum],[x_sum,y_sum,len(x)]])
    b_matrix = np.array([x_z_sum,y_z_sum,z_sum])
    coefficients = np.linalg.solve(a_matrix,b_matrix)
    z_final = (coefficients[0]*x) + (coefficients[1]*y) + coefficients[2]
    plot_standard_least_square(x,y,z,z_final,file)
def plot_total_least_square(x,y,z,z_final,file):
    fig, ax = plt.subplots()
    ax = plt.axes(projection='3d')
    ax.scatter3D(x, y, z, color='g',s=5)
    ax.plot(x, y, z_final, '--r')
    plt.title("Total Least Square Curve for " + file)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    plt.show()
    
def compute_total_least_square(file):
    '''
    Total Least Squares is computed by taking a plane d - ax+by+cz 
    where d corresponds to plane equation considering the means
    '''
    a= np.loadtxt(file, delimiter=",", dtype = float)
    x= a[:,0]
    y = a[:,1]
    z= a[:,2]
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    z_mean = np.mean(z)
    a = np.matrix([x-x_mean,y-y_mean,z-z_mean]).T
    b = np.dot(a.T,a)
    eig_values,eig_vectors = np.linalg.eig(b)
    eig_min_index = np.argmin(eig_values)
    eig_vector_min = np.array(eig_vectors[:,eig_min_index])
    a,b,c= eig_vector_min[0],eig_vector_min[1],eig_vector_min[2]
    d = a*x_mean + b*y_mean + c*z_mean
    z_final = (d- ((a*x)+(b*y)))/c
    plot_total_least_square(x,y,z,z_final,file)
def ransac_TLS(points):
    x= points[:,0]
    y= points[:,1]
    z= points[:,2]
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    z_mean = np.mean(z)
    a = np.matrix([x-x_mean,y-y_mean,z-z_mean]).T
    b = np.dot(a.T,a)
    eig_values,eig_vectors = np.linalg.eig(b)
    eig_min_index = np.argmin(eig_values)
    eig_vector_min = np.array(eig_vectors[:,eig_min_index])
    a,b,c= eig_vector_min[0],eig_vector_min[1],eig_vector_min[2]
    d = a*x_mean + b*y_mean + c*z_mean
    return (a,b,c,d)
def plot_ransac(x,y,z,z_final,file):
    fig, ax = plt.subplots()
    ax = plt.axes(projection='3d')
    ax.scatter3D(x, y, z, color='m',s=5)
    ax.plot(x, y, z_final, '--g')
    plt.title("RANSAC Curve for " + file)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    plt.show()
def compute_ransac(file):
    data= np.loadtxt(file, delimiter=",", dtype = float)
    x_main = data[:,0]
    y_main = data[:,1]
    z_main = data[:,2]
    t = 0.08
    iterations = 1000
    inliers = []
    i = 1
    while (i<iterations):
        random_indices = np.random.choice(data.shape[0], size=3)
        x = x_main[random_indices]
        y = y_main[random_indices]
        z = z_main[random_indices]
        sample_array = np.array([x,y,z]).T
        #Getting the coefficients of the points above using Total Least Square
        a,b,c,d  =  ransac_TLS(sample_array)
        #Calculating the distance of all points from a plane with coefficients dervied above
        distance = (a*x_main + b *y_main + c*z_main+d)/ np.sqrt(a**2 + b**2 + c**2)
        index_threshold = np.where ( np.abs(distance) <= t)[0]
        if (len(index_threshold) > len(inliers)):
            a_final,b_final,c_final,d_final = a,b,c,d
            inliers = index_threshold
        i+=1
    if(c_final !=0):
        z_final = -(a_final*x_main + b_final*y_main + d_final) / c_final
        plot_ransac(x_main,y_main,z_main,z_final,file)
    else : 
        print(a_final,b_final,c_final,d_final)
    return True
def main():
    compute_standard_least_square("pc1.csv")
    compute_standard_least_square("pc2.csv")
    compute_total_least_square("pc1.csv")
    compute_total_least_square("pc2.csv")
    compute_ransac("pc1.csv")
    compute_ransac("pc2.csv")
if __name__ == '__main__':
    main()
