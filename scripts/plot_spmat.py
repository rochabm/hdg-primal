import sys
import numpy as np
import matplotlib.pylab as plt
import scipy.sparse as sps
from scipy.io import *

if __name__ == "__main__":

    A = np.loadtxt(sys.argv[1],skiprows=2)
    IP = np.loadtxt(sys.argv[2],skiprows=4)   
    n = np.shape(A)[0]
    B = np.zeros((n,n))    
    for i in range(n):
        for j in range(n):
            ii = IP[i,1]
            jj = IP[j,1]
            B[i,j] = A[ii,jj]       
   
    Ma = sps.csr_matrix(A)    
    plt.spy(Ma,markersize=0.5)
    plt.savefig("natural.png")
    plt.show()

    Mb = sps.csr_matrix(B)    
    plt.spy(Mb,markersize=0.5)
    plt.savefig("rcm.png")
    plt.show()
    
