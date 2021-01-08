import numpy as np
import function_base as fct
import matplotlib.pyplot as plt


def ResolMCEN(A, b):
    aa = A.T@A
    bb = A.T@b

    l = fct.Cholesky(aa)
    lt = l.T

    y = fct.ResolTriInf(l, bb)
    x = fct.ResolTriSup(lt, y)
    er = np.linalg.norm(A@x-b)
    return x, er


def ResolMCQr(A, b):
    Q, R = DecompositionGS2(A)
    return fct.ResolTriSup(R, Q.T@b)


def ResolMCNP(a, b):
    x = (np.linalg.lstsq(a, b, rcond=None))
    er = np.linalg.norm(a@x[0]-b)
    return x[0], er


def DecompositionGS2(A):
    m, n = A.shape
    v = np.zeros((m, n))
    R = np.zeros((n, n))
    Q = np.zeros((m, n))

    for j in range(0, n):
        v[:, j] = A[:, j]
        for i in range(0, j):
            R[i, j] = np.dot((Q[:, i].conjugate()).T, A[:, j])
            v[:, j] = v[:, j] - R[i, j]*Q[:, i]

        R[j, j] = np.linalg.norm(v[:, j], 2)
        Q[:, j] = v[:, j]/R[j, j]

    return Q, R


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


def cercle():

    donne_x, donne_y = fct.donnees_partie3()

    x = np.array(donne_x)
    y = np.array(donne_y)

    n = x.size
    a = np.ones((n, 3))
    b = np.ones((n, 1))

    a[:, 0] = 2*x
    a[:, 1] = 2*y

    b[:, 0] = x**2+y**2

    result, er = ResolMCEN(a, b)

    alpha, beta, gamma = result

    x0 = alpha
    y0 = beta
    r = np.sqrt(alpha**2 + beta**2 + gamma)
    x1, y1 = draw_circle(x0, y0, r)

    plt.scatter(x1, y1, s=2)
    plt.scatter(x, y, s=20)
    plt.show()


def draw_circle(x0, y0, r):

    x = []
    y = []

    for i in np.arange(-r+x0, r+x0, 0.01):
        for j in np.arange(-r+y0, r+y0, 0.01):
            if r**2 >= pow(i - x0, 2) + pow(j - y0, 2) >= (r ** 2)-0.5:
                x.append(i)
                y.append(j)

    print(x, y)

    return x, y


