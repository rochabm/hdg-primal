import sys
import numpy as np
import matplotlib.pylab as plt
import scipy.sparse as sps
from scipy.io import *

if __name__ == "__main__":

    gk = range(11)
    P2k = np.zeros((11))
    Q2k = np.zeros((11))
    P3k = np.zeros((11))
    Q3k = np.zeros((11))    

    print("\n")
    print("2D elements tri vs quad")
    print("k\t Pk\t Qk\t dif")
    for k in range(11):
        dimpk = (k+1)*(k+2)/2
        dimqk = (k+1)**2
        P2k[k] = dimpk
        Q2k[k] = dimqk
        print("%d\t %d\t %d\t %d" % (k,dimpk,dimqk,dimqk-dimpk))

    print("\n")
    print("3D elements tet vs hex")
    print("k\t Pk\t Qk\t dif")
    for k in range(11):
        dimpk = (k+1)*(k+2)*(k+3)/6
        dimqk = (k+1)**3
        P3k[k] = dimpk
        Q3k[k] = dimqk
        print("%d\t %d\t %d\t %d" % (k,dimpk,dimqk,dimqk-dimpk))
    print("\n")        

    # plota graficos comparativos
    plt.plot(gk,P2k,label="polinomios 2D Pk")
    plt.plot(gk,Q2k,label="polinomios 2D Qk")
    plt.legend(loc="best")
    plt.show()

    plt.plot(gk,P3k,label="polinomios 3D Pk")
    plt.plot(gk,Q3k,label="polinomios 3D Qk")
    plt.legend(loc="best")
    plt.show()

