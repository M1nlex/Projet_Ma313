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

    return x

def ResolMCQr(A,b):
    Q,R = DecompositionGS2(A)
    return fct.ResolTriSup(R,Q.T@b)

def ResolMCNP(A,b):
    pass

def DecompositionGS2(A):
    m,n = A.shape
    v = np.zeros((m,n))
    R = np.zeros((n,n))
    Q = np.zeros((m,n))

    for j in range(0,n):
        v[:,j] = A[:,j]
        for i in range(0,j):
            R[i,j] = np.dot( (Q[:,i].conjugate()).T, A[:,j] )
            v[:,j] = v[:,j] - R[i,j]*Q[:,i]

        R[j,j] = np.linalg.norm(v[:,j],2)
        Q[:,j] = v[:,j]/R[j,j]

    return Q,R
