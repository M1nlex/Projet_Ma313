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
    Q,R = DecompositionGS2(A)
    return fct.ResolTriSup(R,Q.T@b)


def ResolMCNP(A, b):
    x = (np.linalg.lstsq(A, b, rcond=None))
    er = np.linalg.norm(A@x[0]-b)
    return x[0], er


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


def test_minimum(a, b):

    x, er = ResolMCEN(a, b)

    for w in range(0, 10**6):

        x1 = np.zeros((len(x), 1))
        for i in range(len(x)):
            temp = np.random.randn(1)/2000
            x1[i][0] = x[i] + temp

        if np.linalg.norm(x-x1) < 10**-3:
            if np.linalg.norm(a@x-b) >= np.linalg.norm(a@x1-b):
                print('nope')

        print(w)