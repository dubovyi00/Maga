import numpy as np
import matplotlib.pyplot as plt
import json

def func(x, theta):
    return np.array([1.0 * theta[0], 1e-07 * x * theta[1] , x**2 * theta[2]])

def read(filename):
    with open(filename, "r") as f:
        plans = json.load(f)["plans"]
        Xmass = []
        Pmass = []
        for plan in plans:
            Xmass.append(plan["x"])
            Pmass.append(plan["p"])
        return Xmass, Pmass

def make_partM(fx):
    M = np.zeros((len(fx), len(fx)))
    for i in range(len(fx)):
        for j in range(len(fx)):
            M[i][j] = fx[i] * fx[j]
    return M

def makeM(x, p, theta):
    Mmass = []
    for j in range(len(x)):
        M = np.zeros((len(x[j]), len(x[j])))
        for i in range(len(x[j])):
            M += p[j][i] * make_partM(func(x[j][i], theta))
        Mmass.append(M)
    return Mmass

def makeD(M):
    Dmass = []
    for i in range(len(M)):
        Dmass.append(np.linalg.inv(M[i]))
    return Dmass

def D_optim(M):
    buf = []
    for m in M:
        buf.append(np.linalg.det(m))
    print(buf)
    return buf

def A_optim(M):
    buf = []
    for m in M:
        buf.append(np.trace(m))
    print(buf)

def E_optim(M):
    buf = []
    for m in M:
        buf.append(sorted(np.linalg.eig(m)[0])[0])
    print(buf)

def F2_optim(D):
    buf = []
    for d in D:
        buf.append((1.0/(len(d))*np.trace(d**2))**(1/2))
    print(buf)

def L_optim(D):
    buf = []
    for d in D:
        eig = np.linalg.eig(d)[0]
        buf.append(sum((eig - np.average(eig))**2))
    print(buf)

def MV_optim(D):
    buf = []
    for d in D:
        buf.append(max([d[i][i] for i in range(len(d))]))
    print(buf)

def G_optim(D, X, theta):
    buf = []
    for i in range(len(D)):
        buf.append(max([np.dot(np.dot(func(X[i][j],theta),D[i]),func(X[i][j], theta).T) for j in range(len(X[i]))]))
    print(buf)

theta = [1, 1, 1]
x, p = read("data.json")
print(x)
print(p)
M = makeM(x, p , theta)
for Mi in M:
    print(Mi)
D = makeD(M)

print("D")
D_optim(M)
print("A")
A_optim(M)
print("E")
E_optim(M)
print("F2")
F2_optim(D)
print("L")
L_optim(D)
print("MV")
MV_optim(D)
print("G")
G_optim(D, x, theta)

for i in range(len(M)):
    print("M " + str(i + 1))
    print(M[i])
    print("D " + str(i + 1))
    print(D[i])

print("4 пункт")
Xmass = []
Pmass = []
q = []
#тут просто можно менять значения
for i in [i * 0.01 for i in range(1, 50, 1)]:
    Xmass.append([-1, 0 , 1])
    #ВОТ ЗДЕСЬ и все будет хорошо. ну и ограничивать массив в зависимости так, за ограничение не заходила
    Pmass.append([i, 1.0 - 2 * i, i])
    q.append(i)
Mmass = makeM(Xmass, Pmass, theta)
result = D_optim(Mmass)
print("q")
print((result.index(max(result)) + 1) * 0.01)
print("value")
print(max(result))

#отрисовка графика
fig = plt.figure()
plt.plot(q, result)
plt.scatter((result.index(max(result)) + 1) * 0.01, max(result))
plt.title('D - optimality')
plt.ylabel('Criterion value')
plt.xlabel('q')
plt.grid(True)
plt.text((result.index(max(result)) + 1) * 0.01 + 0.01, max(result), 'max', fontsize=12)
plt.show()
