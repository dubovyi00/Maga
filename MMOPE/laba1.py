import numpy as np
import matplotlib.pyplot as plt
import json

class PlanElement(object):
    def __init__(self, x, p):
        self.x = x
        self.p = p

class Model(object):
    def __init__(self, q):
        self.plan = np.array([PlanElement(-1, q), PlanElement(0, 1-2*q), PlanElement(1, q)])
        self.a_optimality(self.plan)

    def a_optimality(self, plan):
        M = np.zeros((2, 2))
        for k in range(len(plan)):
            f = np.array([1, self.plan[k].x ** 2])
            for i in range(2):
                M[i] += self.plan[k].p * f[i] * f
        self.A = np.linalg.inv(M).trace()

# Перемножение вектор функции
def func(x):
    st = np.array([1, x**2]).reshape(1, 2)
    column = np.array([1, x**2]).reshape(2, 1)
    return column * st

# D-оптимальность
def d_optimality(d):
    return np.linalg.det(d)

# A-оптимальность
def a_optimality(d):
    return np.trace(d)

# E-оптимальность
def e_optimality(d):
    return np.max(np.linalg.eig(d)[0])

# Ф-оптимальность
def f_optimality(d):
    p = 2
    return 1.0/p * (D**p).trace()

# Λ-оптимальность
def l_optimality(d):
    return np.sum((np.linalg.eig(d)[0] - np.average(np.linalg.eig(d)[0]))**2)

# MV-оптимальность
def mv_optimality(d):
    return np.max(np.diag(d))

# G-оптимальность
def g_optimality(D, x):
    d = np.zeros(3)
    for i in range(3):
        st = np.array([1, x[i]*x[i]]).reshape(1, 2)
        column = np.array([1, x[i]*x[i]]).reshape(2, 1)
        d[i] = st @ D @ column
    return np.max(d)

# Чтение планов
def read_plans():
    with open("data.json", "r") as f:
        plans = json.load(f)["plans"]
        return plans

plan_num = int(input("Выберите план (от 1 до 4): "))
plans = read_plans()
x = plans[plan_num-1]["x"]
p = plans[plan_num-1]["p"]

# Информационная и дисперсионная матрицы
M = 0
for i in range(3):
    M += p[i] * func(x[i])
D = np.linalg.inv(M)
print("Информационная матрица Фишера:")
print(M)
print("\nДисперсионная матрица:")
print(D)

# Критерии оптимальности
print("\nD-оптимальность:", '%.6f' % d_optimality(D))
print("A-оптимальность:", '%.6f' % a_optimality(D))
print("E-оптимальность:", '%.6f' % e_optimality(D))
print("Ф-оптимальность:", '%.6f' % f_optimality(D))
print("Λ-оптимальность:", '%.6f' % l_optimality(D))
print("MV-оптимальность:", '%.6f' % mv_optimality(D))
print("G-оптимальность:", '%.6f' % g_optimality(D, x))

# Поиск оптимальных значений параметра и критерия А-оптимальности
xPlot, yPlot = [], []
for q in np.arange(0.01, 0.5, 0.01):
    if q not in (-0.5, 0.5, 0):
        xPlot.append(q)
        yPlot.append(Model(q).A)
print("\nОптимальные значения, найденные по графику:")
print("q:", '%.6f' % xPlot[yPlot.index(min(yPlot))])
print("A-optimality:", '%.6f' % min(yPlot))

# Построение графика изменения критерия А-оптимальности
fig = plt.figure()
plt.plot(xPlot, yPlot)
plt.scatter((yPlot.index(min(yPlot)) + 1) * 0.01, min(yPlot))
plt.ylabel('A-optimality')
plt.xlabel('q')
plt.grid(True)
plt.text((yPlot.index(min(yPlot)) + 1) * 0.01 + 0.01, min(yPlot), 'min')
plt.show()