import math
import json
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

m = 6
gamma = 2

# f(x) - это
def f(a, b):
    f = np.array([[1], [a], [b], [a * b], [a ** 2], [b ** 2]])
    return f

# чтение плана из файла
def read_plan():
    x1 = []
    x2 = []
    p = []
    with open("data2.json", "r") as f:
        plans = json.load(f)["plan"]
        for plan in plans:
            x1.append(plan["x1"])
            x2.append(plan["x2"])
            p.append(plan["p"])
        return x1, x2, p

# Построение матрицы M
def calc_M(c, d, p1, m):
    M1 = np.zeros((m, m))
    n = len(x1)
    for q in range(0, n - 1):
        M1 += p1[q] * f(c[q], d[q]) @ np.transpose(f(c[q], d[q]))
    return M1

# Построение дисперсионной матрицы D
def calc_D(M1):
    D1 = np.linalg.inv(M1)
    return D1

# 2. найти глобальный экстремум: xs = arg max phi(x, es)
def GlobExtr_2(D):
    phiM = -100
    xMax1 = 0
    xMax2 = 0
    l1 = -1
    i1 = -1
    while (i1 <= 1.0):
        i2 = -1
        while (i2 <= 1.0):
            phi = np.transpose(f(i1, i2)) @ D @ D @ f(i1, i2) # A-plan
            if (phiM <= phi):
                phiM = phi
                xMax1 = i1
                xMax2 = i2
                l1 += 1
            i2 += 0.01
        i1 += 0.01
    return phiM, xMax1, xMax2

# 3. проверка необходимых и достаточных условий
def Usl_3(phiM, x1M, x2M, a, b, M, D):
    delta = np.abs(phiM) * 0.01
    print("delta = ", delta)
    d = np.abs(np.trace(D @ D @ M) - phiM) # A-plan
    print("d = ", d)
    alg = 1 if d <= delta else 0
    return x1M, x2M, a, b, alg

# 4. составление нового плана
def NewPlan_4(i1, i2, ps1, b1, b2, a1):
    ns = len(ps1)
    for w in range(0, ns):
        ps1[w] = (1 - a1) * ps1[w]
    
    b1.append(i1)
    b2.append(i2)
    ps1 = np.append(ps1, a1)

    print("len(x1) = ", len(b1), "x1:")
    print("len(x2) = ", len(b2), "x2:")
    print("len(p) = ", len(ps1), "p:")
    return ps1, b1, b2

# Создание графика
def DrawGraph(a, b):
    for i1 in range(0, len(a)):
        plt.scatter(a[i1], b[i1])
    plt.plot()
    plt.show()


x1, x2, p = read_plan()

# 1. выбор плана e0, s = 0
s = 0
print("План e", s)
M = []
D = []

while (True):
    # 2 - глобальный экстремум
    M = calc_M(x1, x2, p, m)
    D = calc_D(M)

    phiMax, x1Max, x2Max = GlobExtr_2(D)

    # 3 - проверка условий -> alg opt
    x1Max, x2Max, x1, x2, alg = Usl_3(phiMax, x1Max, x2Max, x1, x2, M, D)
    
    if (alg == 1):
        print("alg opt")
        print("x1 = ", x1)
        print("x2 = ", x2)
        print("p = ", p)
        break

    # 4 - новый план
    a = 1 / (len(p))
    p, x1, x2 = NewPlan_4(x1Max, x2Max, p, x1, x2, a)

    # 5 - сравнение функционала
    Ms = calc_M(x1, x2, p, m)
    Ds = calc_D(Ms)
    while (True):
        if np.trace(Ds) > np.trace(D):
            print("det(Ms) = ", np.linalg.det(Ms), "det(M) = ", np.linalg.det(M))
            a /= gamma
            print("a = ", a)
            p, x1, x2 = NewPlan_4(x1Max, x2Max, p, x1, x2, a)
            Ms = calc_M(x1, x2, p, m)
            Ds = calc_D(Ms)
        else:
            break
    
    s += 1
    M = calc_M(x1, x2, p, m)
    D = calc_D(M)
    print("s = ", s)
    if (s == 1 or s % 250 == 0):
        print("len(x1) = ", len(x1), "x1:")
        print(x1)
        print("len(x2) = ", len(x2), "x2:")
        print(x2)
        print("len(p) = ", len(p), "p:")
        print(p)
        print("Проверка на оптимальность - неочищенный план A")
        print("Правая часть: ", np.trace(D))
        print("Левая часть: ", np.trace(D @ D @ M))

M = calc_M(x1, x2, p, m)
D = calc_D(M)

#Проверка на оптимальность
print("Первая проверка на оптимальность - неочищенный план A")
print("Правая часть: ", np.trace(D))
print("Левая часть: ", np.trace(D @ D @ M))

newX1 = [0]
newX2 = [0]
newP = [0]
newX1[0] = x1[0]
newX2[0] = x2[0]
newP[0] = p[0]
mu = 0.1
pi = 0.02
k = len(x1)
r = 0
i = 1
print("sum p = ", np.sum(p))

while (True):
    if r >= len(p) - 1:
        break
        
    while (i < len(p)):
        scal = (x1[r] - x1[i])**2 + (x2[r] - x2[i])**2 #скаляр
        if scal <= mu: #проверка на близость
            p[r] += p[i] #добавление веса к точке
            p = np.delete(p, i)
            x1 = np.delete(x1, i)
            x2 = np.delete(x2, i)
            i -= 1
        i += 1
    r += 1
    i = r + 1

print("Выполняется очистка плана")

print("len(x1) = ", len(x1), "x1:")
print(x1)
print("len(x2) = ", len(x2), "x2:")
print(x2)
print("len(p) = ", len(p), "p:")
print(p)

sumP = 0
newX1 = x1
newX2 = x2
newP = p
for i in range(1, len(newP)): #найти в массиве р малые веса
    if (newP[i] <= pi):
        sumP += newP[i] #сумма малых весов
        newP[i] = 0

newX11 = []
newX21 = []
newP1 = []
for i in range(0, len(newP)): #удаление из массивов элементов с нулевыми весами
    if newP[i] != 0:
        newP1.append(newP[i])
        newX11.append(newX1[i])
        newX21.append(newX2[i])

p = newP1
x1 = newX11
x2 = newX21

sumP = sumP/len(p)

print("План очищен!")
M = calc_M(x1, x2, p, m)
D = calc_D(M)

#Проверка на оптимальность
print("Вторая проверка на оптимальность - очищенный план A")
print("Правая часть: ", np.trace(D))
print("Левая часть: ", np.trace(D @ D @ M))
DrawGraph(x1, x2)

print("len(x1) = ", len(x1), "x1:")
print(x1)
print("len(x2) = ", len(x2), "x2:")
print(x2)
print("len(p) = ", len(p), "p:")
print(p)
