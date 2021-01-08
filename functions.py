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
    return x[0], float(x[1]**(1/2))


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


def test_minimum(a, b, nbr_tests = 10**6):

    x1, er1 = ResolMCEN(a, b)
    x2 = ResolMCQr(a, b)
    x3, er3 = ResolMCNP(a, b)

    listeRes1 = []
    listeRes2 = []
    listeRes3 = []
    for w in range(0, nbr_tests):

        X1 = np.zeros((len(x1), 1))
        X2 = np.zeros((len(x1), 1))
        X3 = np.zeros((len(x1), 1))

        for i in range(len(x1)):
            temp = np.random.randn(1)/100000
            X1[i][0] = x1[i] + temp
            X2[i][0] = x2[i] + temp
            X3[i][0] = x3[i] + temp

        if np.linalg.norm(x1-X1) < 10**-3:
            if np.linalg.norm(a@x1-b) >= np.linalg.norm(a@X1-b):
                listeRes1.append(0)
            else:
                listeRes1.append(1)
        else:
            listeRes1.append(-1)

        if np.linalg.norm(x2-X2) < 10**-3:
            if np.linalg.norm(a@x2-b) >= np.linalg.norm(a@X2-b):
                listeRes2.append(0)
            else:
                listeRes2.append(1)
        else:
            listeRes2.append(-1)

        if np.linalg.norm(x3-X3) < 10**-3:
            if np.linalg.norm(a@x3-b) >= np.linalg.norm(a@X3-b):
                listeRes3.append(0)
            else:
                listeRes3.append(1)
        else:
            listeRes3.append(-1)

        #print(w)
    liste_abscisse = range(0, nbr_tests)

    plt.scatter(liste_abscisse,listeRes1,label="équations normales")
    plt.scatter(liste_abscisse,listeRes2,label="décomposition QR")
    plt.scatter(liste_abscisse,listeRes3,label="méthode numpy")
    plt.legend()
    plt.title("Vérification de la justesse des résultats")
    plt.show()


def cercle(methode=0):

    donne_x, donne_y = fct.donnees_partie3()

    x = np.array(donne_x)
    y = np.array(donne_y)

    n = x.size
    a = np.ones((n, 3))
    b = np.ones((n, 1))

    a[:, 0] = 2*x
    a[:, 1] = 2*y

    b[:, 0] = x**2+y**2

    if methode == 0:
        result, er = ResolMCEN(a, b)
        alpha1, beta1, gamma1 = result
        x01 = alpha1
        y01 = beta1
        r1 = np.sqrt(alpha1 ** 2 + beta1 ** 2 + gamma1)
        x1, y1 = draw_circle(x01, y01, r1)
        plt.scatter(x1, y1, s=2, label='ResolMCEN')
    if methode == 1:
        result = ResolMCQr(a, b)
        alpha2, beta2, gamma2 = result
        x02 = alpha2
        y02 = beta2
        r2 = np.sqrt(alpha2 ** 2 + beta2 ** 2 + gamma2)
        x2, y2 = draw_circle(x02, y02, r2)
        plt.scatter(x2, y2, s=2, label='ResolMCQr')
    if methode == 3:
        result, er = ResolMCNP(a, b)
        alpha3, beta3, gamma3 = result
        x03 = alpha3
        y03 = beta3
        r3 = np.sqrt(alpha3 ** 2 + beta3 ** 2 + gamma3)
        x3, y3 = draw_circle(x03, y03, r3)
        plt.scatter(x3, y3, s=2, label='ResolMCNP')

    plt.scatter(x, y, s=20)
    plt.legend()
    plt.show()


def draw_circle(x0, y0, r):

    x = []
    y = []

    for i in np.arange(-r+x0, r+x0, 0.01):
        for j in np.arange(-r+y0, r+y0, 0.01):
            if r**2 >= pow(i - x0, 2) + pow(j - y0, 2) >= (r ** 2)-0.3:
                x.append(i)
                y.append(j)

    return x, y
