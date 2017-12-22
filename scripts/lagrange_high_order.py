import sys
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def mfind(x3d,val):
    m,n,l = np.shape(x3d)
    for i in range(m):
        for j in range(n):
            for k in range(l):
                if(x3d[i,j,k]==val):
                    return (i,j,k)

    print("erro mfind %d not found" % val)
    sys.exit(1)

def mfind2(x2d,val):
    m,n = np.shape(x2d)
    for i in range(m):
        for j in range(n):
            if(x2d[i,j]==val):
                return (i,j)

    print("erro mfind %d not found" % val)
    sys.exit(1)

def find_coord_id(num,X,Y,Z,a,b,c):
    nx,ny,nz = np.shape(num)
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if (X[i,j,k]==a and Y[i,j,k]==b and Z[i,j,k]==c):
                    return int(num[i,j,k])
    return -1

def show_elem(num,X,Y,Z):
    nx,ny,nz = np.shape(num)
    print("\nNumbering")
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                print("%d %d %d ind %3d pos %6.3f %6.3f %6.3f" %
                      (i+1,j+1,k+1,num[i,j,k]+1,X[i,j,k],Y[i,j,k],Z[i,j,k]))

def register(num,i,j,k,ind):
    if(num[i,j,k] == -1):
        print("alterando %d %d %d para %d" % (i,j,k,ind))
        numbr[i,j,k] = ind
    else:
        print("erro numbr %d %d %d = ind %d -> old %d" % (i,j,k,ind,num[i,j,k]))
        sys.exit(1)

def plot_edge(na,nb,ax):
    ia,ja,ka = mfind(numbr, na)
    ib,jb,kb = mfind(numbr, nb)
    xa,ya,za = X[ia,ja,ka],Y[ia,ja,ka],Z[ia,ja,ka]
    xb,yb,zb = X[ib,jb,kb],Y[ib,jb,kb],Z[ib,jb,kb]
    ax.plot([xa,xb],[ya,yb],[za,zb], c='k')

def gen_code(numbr,X,Y,Z):
    gen_inod(numbr,X,Y,Z)
    gen_face(numbr,X,Y,Z)

def gen_inod(numbr,X,Y,Z):
    nx,ny,nz = np.shape(numbr)
    nen = nx*ny*nz

    print("\n\ninod information\n\n")
    print("if(nen.eq.%d) then" % nen)
    print("c")
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                ind = numbr[i,j,k]
                print("   inod(%1d,%1d,%1d) = %d" % (i+1,j+1,k+1,ind+1))
        print("c")
    print("end if")

def gen_face(numbr,X,Y,Z):
    nx,ny,nz = np.shape(numbr)
    nen = nx*ny*nz

    # nodes
    x1d = np.linspace(-1.0, 1.0, nx)
    x1d = np.roll(x1d,1)
    x1d[0],x1d[1] = x1d[1],x1d[0]
    print("1D Nodes")
    print(x1d[:])

    edges = np.array([[0,1],[1,2],[3,2],[0,3]])

    # numbering array
    numfc = np.zeros((nx,ny))
    numfc[:,:] = -1
    xi  = np.zeros((nx,ny))
    eta = np.zeros((nx,ny))

    # create node positions
    for j in range(ny):
        for i in range(nx):
            xxi,xxe = x1d[i],x1d[j]
            xi[i,j],eta[i,j] = xxi,xxe

    # order them
    ind=0
    for j in range(2):
        for i in range(2):
            if (i == 0 and j == 1):
                ind = ind + 1
                numfc[i, j] = ind
                ind = ind - 1
            elif (i == 1 and j == 1):
                ind = ind - 1
                numfc[i, j] = ind
                ind = ind + 1
            else:
                numfc[i, j] = ind
            ind = ind + 1

    m = ny-2
    if(m>0):
        # baixo
        j=0
        for i in range(2,nx):
            numfc[i,j] = ind
            ind = ind + 1
        # dir
        i = 1
        for j in range(2, ny):
            numfc[i, j] = ind
            ind = ind + 1
        # cima
        j=1
        for i in range(ny-1,1,-1):
            numfc[i, j] = ind
            ind = ind + 1
        # esq
        i=0
        for j in range(nx-1,1,-1):
            numfc[i, j] = ind
            ind = ind + 1

        # interior
        for j in range(2,ny):
            for i in range(2,nx):
                numfc[i,j] = ind
                ind = ind +1

    n = nx*ny
    print("\n\nidside information\n\n")
    print("subroutine shlhxbk (elem) e shlhxpbk (face)")
    print("if(nen.eq.%d) then" % nen)
    print("c")
    # face frente y=-1
    for i in range(n):
        ii, jj = mfind2(numfc, i)
        #print(ii,jj)
        xxi,xeta = xi[ii,jj], eta[ii,jj]
        nidx = find_coord_id(numbr,X,Y,Z,xxi,-1.0,xeta)
        print("   idside(%1d,%1d) =  %d" % (1, i+1, nidx+1))
    print("c")
    # face esq x=-1
    for i in range(n):
        ii, jj = mfind2(numfc, i)
        xxi,xeta = xi[ii,jj], eta[ii,jj]
        nidx = find_coord_id(numbr,X,Y,Z,-1.0,xxi,xeta)
        print("   idside(%1d,%1d) =  %d" % (2, i+1, nidx+1))
    print("c")
    # face dir x=+1
    for i in range(n):
        ii, jj = mfind2(numfc, i)
        xxi,xeta = xi[ii,jj], eta[ii,jj]
        nidx = find_coord_id(numbr,X,Y,Z,+1.0,xxi,xeta)
        print("   idside(%1d,%1d) =  %d" % (3, i+1, nidx+1))
    print("c")
    # face tras y=+1
    for i in range(n):
        ii, jj = mfind2(numfc, i)
        xxi,xeta = xi[ii,jj], eta[ii,jj]
        nidx = find_coord_id(numbr,X,Y,Z,xxi,+1.0,xeta)
        print("   idside(%1d,%1d) =  %d" % (4, i+1, nidx+1))
    print("c")
    # face baixo z=-1
    for i in range(n):
        ii, jj = mfind2(numfc, i)
        xxi,xeta = xi[ii,jj], eta[ii,jj]
        nidx = find_coord_id(numbr,X,Y,Z,xxi,xeta,-1.0)
        print("   idside(%1d,%1d) =  %d" % (5, i+1, nidx+1))
    print("c")
    # face cima z=+1
    for i in range(n):
        ii, jj = mfind2(numfc, i)
        xxi,xeta = xi[ii,jj], eta[ii,jj]
        nidx = find_coord_id(numbr,X,Y,Z,xxi,xeta,+1.0)
        print("   idside(%1d,%1d) =  %d" % (6, i+1, nidx+1))
    print("c")
    print("end if")

if __name__ == "__main__":

    # define polynomial order
    p = 1
    
    if(len(sys.argv) > 1):
        p = int(sys.argv[1])

    # configure plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # some parameters
    n_nodes = (p+1)**3
    nx,ny,nz = p+1,p+1,p+1

    # nodes
    x1d = np.linspace(-1.0, 1.0, nx)
    x1d = np.roll(x1d,1)
    x1d[0],x1d[1] = x1d[1],x1d[0]
    print("1D Nodes")
    print(x1d[:])

    # edge (bottom, top)
    bedges = np.array([[0,1],[1,2],[3,2],[0,3]])
    tedges = np.array([[4,5],[5,6],[7,6],[4,7]])

    # numbering array
    numbr = np.zeros((nx,ny,nz))
    numbr[:,:,:] = -1
    X = np.zeros((nx,ny,nz))
    Y = np.zeros((nx,ny,nz))
    Z = np.zeros((nx,ny,nz))    

    # create node positions
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                xx,yy,zz = x1d[i],x1d[j],x1d[k]
                X[i,j,k],Y[i,j,k],Z[i,j,k] = xx,yy,zz
                
    ### number hex vertices (8-vertices)
    ind = 0
    for k in range(2):
        for j in range(2):
            for i in range(2):
                xx,yy,zz = x1d[i], x1d[j], x1d[k]
                if(i==0 and j==1):
                    ind = ind + 1
                    numbr[i,j,k] = ind
                    ax.scatter(xx,yy,zz,c='b', marker='o')
                    ax.text(xx,yy,zz,ind+1,size=10,zorder=1,color='gray')
                    ind = ind - 1                
                elif(i==1 and j==1):
                    ind = ind - 1
                    numbr[i,j,k] = ind
                    ax.scatter(xx,yy,zz,c='b', marker='o')
                    ax.text(xx,yy,zz,ind+1,size=10,zorder=1,color='gray')
                    ind = ind + 1
                else:
                    numbr[i,j,k] = ind
                    ax.scatter(xx,yy,zz,c='b', marker='o')
                    ax.text(xx,yy,zz,ind+1,size=10,zorder=1,color='gray')
                ind = ind + 1

    # mid-edge - bottom
    nme = nx-2
    for i in range(4):
        a = bedges[i][0]
        b = bedges[i][1]        
        ia,ja,ka = mfind(numbr, a)
        ib,jb,kb = mfind(numbr, b)        
        xa,ya,za = X[ia,ja,ka],Y[ia,ja,ka],Z[ia,ja,ka]
        xb,yb,zb = X[ib,jb,kb],Y[ib,jb,kb],Z[ib,jb,kb]
        plot_edge(a, b, ax)
        # continue numbering
        xx = np.linspace(-1,1,nx)
        xx = xx[1:-1]
        if(i>=2): xx = xx[::-1]
        ii,jj=1,1
        for e in range(nme):
            dx, dy, dz = int((xb-xa)/2), int((yb-ya)/2), int((zb-za)/2)
            if(dx==1):
                register(numbr,ii+dx,ja,ka,ind)
                ax.scatter(xx[e],ya,za,c='c', marker='o')
                ax.text(xx[e],ya,za,ind+1,size=10,zorder=1,color='gray')
            elif(dy==1):
                register(numbr,ia,jj+dy,ka,ind)
                ax.scatter(xa,xx[e],za,c='c', marker='o')
                ax.text(xa,xx[e],za,ind+1,size=10,zorder=1,color='gray')
            ii+=1
            jj+=1
            ind+=1
            
    ### mid-edge - top
    nme = nx-2
    for i in range(4):
        a = tedges[i][0]
        b = tedges[i][1]
        ia,ja,ka = mfind(numbr, a)
        ib,jb,kb = mfind(numbr, b)
        xa,ya,za = X[ia,ja,ka],Y[ia,ja,ka],Z[ia,ja,ka]
        xb,yb,zb = X[ib,jb,kb],Y[ib,jb,kb],Z[ib,jb,kb]
        plot_edge(a,b,ax)
        # continue numbering
        xx = np.linspace(-1,1,nx)
        xx = xx[1:-1]
        if(i>=2): xx = xx[::-1]
        ii,jj=1,1
        for e in range(nme):
            dx, dy, dz = int((xb-xa)/2), int((yb-ya)/2), int((zb-za)/2)
            if(dx==1):
                register(numbr,ii+dx,ja,ka,ind)
                ax.scatter(xx[e],ya,za,c='c', marker='o')
                ax.text(xx[e],ya,za,ind+1,size=10,zorder=1,color='gray')
            elif(dy==1):
                register(numbr,ia,jj+dy,ka,ind)
                ax.scatter(xa,xx[e],za,c='c', marker='o')
                ax.text(xa,xx[e],za,ind+1,size=10,zorder=1,color='gray')
            ii+=1
            jj+=1
            ind+=1
               
    # plot other edges
    plot_edge(0, 4, ax)
    plot_edge(1, 5, ax)
    plot_edge(2, 6, ax)
    plot_edge(3, 7, ax)
   
    ### middle - borders
    for k in range(2,nz):
        for j in range(2):
            for i in range(2):
                if(i==0 and j==1):
                    ind = ind + 1
                    numbr[i,j,k] = ind
                    xx,yy,zz = X[i,j,k],Y[i,j,k],Z[i,j,k]
                    ax.scatter(xx,yy,zz,c='c', marker='o')
                    ax.text(xx,yy,zz,ind+1,size=10,zorder=1,color='gray')
                    ind = ind - 1                
                elif(i==1 and j==1):
                    ind = ind - 1
                    numbr[i,j,k] = ind
                    xx,yy,zz = X[i,j,k],Y[i,j,k],Z[i,j,k]
                    ax.scatter(xx,yy,zz,c='c', marker='o')
                    ax.text(xx,yy,zz,ind+1,size=10,zorder=1,color='gray')
                    ind = ind + 1                    
                else:
                    xx,yy,zz = X[i,j,k],Y[i,j,k],Z[i,j,k]
                    ax.scatter(xx,yy,zz,c='c', marker='o')
                    ax.text(xx,yy,zz,ind+1,size=10,zorder=1,color='gray')
                    numbr[i,j,k] = ind
                ind = ind + 1

                
    ### middle - mid-egdes
    nme = nx-2
    if(nme>0):

        for k in range(2,nz):
            # front
            j=0
            for i in range(2,nx):
                numbr[i,j,k] = ind
                xx,yy,zz = X[i,j,k],Y[i,j,k],Z[i,j,k]
                ax.scatter(xx,yy,zz,c='c', marker='o')
                ax.text(xx,yy,zz,ind+1,size=10,zorder=1,color='gray')
                ind = ind + 1
            # right
            i=1
            for j in range(2,ny):
                numbr[i,j,k] = ind
                xx,yy,zz = X[i,j,k],Y[i,j,k],Z[i,j,k]
                ax.scatter(xx,yy,zz,c='c', marker='o')
                ax.text(xx,yy,zz,ind+1,size=10,zorder=1,color='gray')
                ind = ind + 1
            # back
            j=1
            for i in range(nx-1,1,-1):
                numbr[i,j,k] = ind
                xx,yy,zz = X[i,j,k],Y[i,j,k],Z[i,j,k]
                ax.scatter(xx,yy,zz,c='c', marker='o')
                ax.text(xx,yy,zz,ind+1,size=10,zorder=1,color='gray')
                ind = ind + 1
            # left
            i=0
            for j in range(ny-1,1,-1):
                numbr[i,j,k] = ind
                xx,yy,zz = X[i,j,k],Y[i,j,k],Z[i,j,k]
                ax.scatter(xx,yy,zz,c='c', marker='o')
                ax.text(xx,yy,zz,ind+1,size=10,zorder=1,color='gray')
                ind = ind + 1

        # bottom
        k = 0
        for j in range(2,ny):
            for i in range(2,nx):
                numbr[i,j,k] = ind
                xx,yy,zz = X[i,j,k],Y[i,j,k],Z[i,j,k]
                ax.scatter(xx,yy,zz,c='c', marker='o')
                ax.text(xx,yy,zz,ind+1,size=10,zorder=1,color='gray')
                ind = ind + 1

        # top
        k = 1
        for j in range(2,ny):
            for i in range(2,nx):
                numbr[i,j,k] = ind
                xx,yy,zz = X[i,j,k],Y[i,j,k],Z[i,j,k]
                ax.scatter(xx,yy,zz,c='c', marker='o')
                ax.text(xx,yy,zz,ind+1,size=10,zorder=1,color='gray')
                ind = ind + 1
                                       
    ### interiors
    for k in range(2,nz):
        for j in range(2,ny):
            for i in range(2,nx):
                numbr[i,j,k] = ind
                xx,yy,zz = X[i,j,k],Y[i,j,k],Z[i,j,k]
                ax.scatter(xx,yy,zz,c='y', marker='o')
                ax.text(xx,yy,zz,ind+1,size=10,zorder=1,color='gray')
                ind = ind + 1

    # done and check if all is OK
    b = np.where(numbr==-1)
    if(np.size(b) != 0):
        print("Error: elemento nao foi criado corretamente")
        sys.exit(1)
    
    # show all the information    
    show_elem(numbr,X,Y,Z)

    # generate fortran77 code
    gen_code(numbr,X,Y,Z)

    plt.show()

