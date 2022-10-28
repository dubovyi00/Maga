import json
import numpy as np
import matplotlib.pyplot as plt

m = 6
gamma = 2


# f(x)
def f(a, b):
    f_ = np.array([[1], [a], [b], [a * b], [a ** 2], [b ** 2]])
    return f_


# чтение плана из файла
def read_plan():
    with open("data2.json", "r") as f:
        x1_ = []
        x2_ = []
        p = []
        plans = json.load(f)["plan"]
        for plan in plans:
            x1_.append(plan["x1"])
            x2_.append(plan["x2"])
            p.append(plan["p"])
        return x1_, x2_, p


# Построение матрицы M
def get_mat_m(c, d, p1, mat_m):
    mat_m = np.zeros((mat_m, mat_m))
    n = len(x1)
    for q in range(0, n - 1):
        mat_m += p1[q] * f(c[q], d[q]) @ np.transpose(f(c[q], d[q]))
    return mat_m


# Построение дисперсионной матрицы D
def get_mat_d(mat_m):
    mat_d = np.linalg.inv(mat_m)
    return mat_d


# 2. найти глобальный экстремум
def get_global_max(D):
    phi_max = -100
    argmax_x1 = 0
    argmax_x2 = 0
    l1 = -1
    i1 = -1
    while i1 <= 1.0:
        i2 = -1
        while i2 <= 1.0:
            phi = (np.transpose(f(i1, i2)) @ D @ D @ f(i1, i2)).item()  # A-plan
            if phi_max < phi:
                phi_max = phi
                argmax_x1 = i1
                argmax_x2 = i2
                l1 += 1
            i2 += 0.01
        i1 += 0.01
    return phi_max, argmax_x1, argmax_x2


# 3. проверка необходимых и достаточных условий
def check_condition(phiM, M, D):
    delta = np.abs(phiM) * 0.01
    print("delta = ", delta)
    d = np.abs(np.trace(D @ D @ M) - phiM)  # A-plan
    print("d = ", d)
    return d <= delta


# 4. составление нового плана
def create_plan(i1, i2, ps1, b1, b2, a1):
    ns = len(ps1)
    for w in range(0, ns):
        ps1[w] = (1 - a1) * ps1[w]

    b1.append(i1)
    b2.append(i2)
    ps1 = np.append(ps1, a1)

    print("len(x1) = ", len(b1))
    print("len(x2) = ", len(b2))
    print("len(p) = ", len(ps1))
    return ps1, b1, b2


# Создание графика
def show_scatter(a, b):
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

while True:
    # 2 - глобальный экстремум
    M = get_mat_m(x1, x2, p, m)
    D = get_mat_d(M)

    phiMax, x1Max, x2Max = get_global_max(D)

    # 3 - проверка условий
    if check_condition(phiMax, M, D):
        print("алгоритм оптимален")
        print("x1 = ", x1)
        print("x2 = ", x2)
        print("p = ", p)
        break

    # 4 - новый план
    a = 1 / len(p)
    p, x1, x2 = create_plan(x1Max, x2Max, p, x1, x2, a)

    # 5 - сравнение функционала
    Ms = get_mat_m(x1, x2, p, m)
    Ds = get_mat_d(Ms)
    while True:
        if np.trace(Ds) > np.trace(D):
            print("det(Ms) = ", np.linalg.det(Ms), "det(M) = ", np.linalg.det(M))
            a /= gamma
            print("a = ", a)
            p, x1, x2 = create_plan(x1Max, x2Max, p, x1, x2, a)
            Ms = get_mat_m(x1, x2, p, m)
            Ds = get_mat_d(Ms)
        else:
            break

    M = get_mat_m(x1, x2, p, m)
    D = get_mat_d(M)
    print("s = ", s)
    if s % 250 == 0:
        print("len(x1) = ", len(x1), "x1:\n", x1)
        print("len(x2) = ", len(x2), "x2:\n", x2)
        print("len(p) = ", len(p), "p:\n", p)
        print("Проверка на оптимальность - неочищенный план A")
        print("Правая часть: ", np.trace(D))
        print("Левая часть: ", np.trace(D @ D @ M))

    s += 1

M = get_mat_m(x1, x2, p, m)
D = get_mat_d(M)

# Проверка на оптимальность
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

while True:
    if r >= len(p) - 1:
        break

    while i < len(p):
        scalar = (x1[r] - x1[i]) ** 2 + (x2[r] - x2[i]) ** 2  # скаляр
        if scalar <= mu:  # проверка на близость
            p[r] += p[i]  # добавление веса к точке
            p = np.delete(p, i)
            x1 = np.delete(x1, i)
            x2 = np.delete(x2, i)
            i -= 1
        i += 1
    r += 1
    i = r + 1

print("Выполняется очистка плана")

print("len(x1) = ", len(x1), "x1:\n", x1)
print("len(x2) = ", len(x2), "x2:\n", x2)
print("len(p) = ", len(p), "p:\n", p)

sumP = 0
newX1 = x1
newX2 = x2
newP = p
for i in range(1, len(newP)):  # найти в массиве р малые веса
    if newP[i] <= pi:
        sumP += newP[i]  # сумма малых весов
        newP[i] = 0

newX11 = []
newX21 = []
newP1 = []
for i in range(0, len(newP)):  # удаление из массивов элементов с нулевыми весами
    if newP[i] != 0:
        newP1.append(newP[i])
        newX11.append(newX1[i])
        newX21.append(newX2[i])

p = newP1
x1 = newX11
x2 = newX21

sumP = sumP / len(p)

print("План очищен!")
M = get_mat_m(x1, x2, p, m)
D = get_mat_d(M)

# Проверка на оптимальность
print("Вторая проверка на оптимальность - очищенный план A")
print("Правая часть: ", np.trace(D))
print("Левая часть: ", np.trace(D @ D @ M))
show_scatter(x1, x2)


print("len(x1) = ", len(x1), "x1:\n", x1)
print("len(x2) = ", len(x2), "x2:\n", x2)
print("len(p) = ", len(p), "p:\n", p)
