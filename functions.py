import numpy as np
import function_base as fct
import matplotlib.pyplot as plt


def ResolMCEN(A, b):
    aa = A.T@A
    bb = A.T@b
    #return np.linalg.solve(aa,bb)

    l = fct.Cholesky(aa)
    lt = l.T

    y = fct.ResolTriInf(l, bb)
    x = fct.ResolTriSup(lt, y)
    er = np.linalg.norm(A@x-b)
    return x, er




def ResolMCQr(A,b):
    pass


def ResolMCNP(A, b):
    x = (np.linalg.lstsq(A, b, rcond=None))
    return x[0], x[1]

def DecompositionGS(A):
    """ Calcul de la décomposition QR de A une matrice carrée.
    L'algorithme de Gram-Schmidt est utilisé.
    La fonction renvoit (Q,R) """
    n,m=A.shape
    if n < m :
        raise Exception('Matrice mal dimensionnée')

    Q=np.zeros((n,m))
    R=np.zeros((n,m))
    for j in range(n):
        for i in range(j):
            R[i,j]=Q[:,i]@A[:,j]
        w=A[:,j]
        for k in range(j):
            w=w-R[k,j]*Q[:,k]
        norme=np.linalg.norm(w)
        if norme ==0:
            raise Exception('Matrice non inversible')
        R[j,j]=norme
        Q[:,j]=w/norme
    return Q,R
