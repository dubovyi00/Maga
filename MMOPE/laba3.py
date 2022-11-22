import sys

import numpy as np
import matplotlib.pyplot as plt

# определение f(x)
def f(a, b):
    f = np.array([[1], [a], [b], [a * b], [a ** 2], [b ** 2]])
    return f

# построение информационной матрицы М
def calc_M(c, d, p1, m):
    M1 = np.zeros((m, m))
    n = len(c)
    for q in range(0, n - 1):
        M1 += p1[q] * f(c[q], d[q]) @ np.transpose(f(c[q], d[q]))
    return M1

# построение дисперсионной матрицы D
def calc_D(M1):
    D1 = np.linalg.inv(M1)
    return D1

# поиск d(x, e)
def calc_d(x1, x2, D1):
    d1 = np.transpose(f(x1, x2)) @ D1 @ f(x1, x2)
    return d1

# отрисовка графика
def draw_graph(a, b):
    for i1 in range(0, len(a)):
        plt.scatter(a[i1], b[i1]) 
    plt.plot()
    plt.show()

# параметры генератора сетки
plan = []
grid = 20 # 10 20
grid_step = {
    10: 0.2,
    20: 0.1
}

N = int(sys.argv[1])
m = 6 
a = -1
plan.append(a)
for i in range(0, grid):
    a += grid_step[grid]
    plan.append(a)

# выбор невырожденного плана 
s = 0
en_x1 = np.random.choice(plan, N, replace=True, p=None)
en_x2 = np.random.choice(plan, N, replace=True, p=None)
print("en_x1 = ", en_x1)
print("en_x2 = ", en_x2)
draw_graph(en_x1, en_x2)

while True:
    print("s = ", s)
    print("plan = ", plan)
    print("en_x1 = ", en_x1)
    print("en_x2 = ", en_x2)
    pi = 1 / len(en_x1)
    p = []
    for i in range(0, len(en_x1)):
        p.append(pi)
    print("p = ", p)
    print("sum(p) = ", np.sum(p))
    # выбор точки x_s 

    M = calc_M(en_x1, en_x2, p, m)
    D = calc_D(M)
    delta = grid_step[grid]
    maxd = -100
    k = 0
    x1_max = -2
    x2_max = -2
    xs_1 = -1
    while (xs_1 <= 1):
        xs_2 = -1
        while (xs_2 <= 1):
            for i in range(0, len(en_x1)):
                d = calc_d(xs_1, xs_2, D)
                if maxd <= d:
                    maxd = d
                    x1_max = xs_1
                    x2_max = xs_2
            xs_2 += delta
        xs_1 += delta

    print("maxd = ", maxd)
    print("x1_max = ", x1_max)
    print("x2_max = ", x2_max)

    # точка x_s добавляется в план
    en1_x1 = np.append(en_x1, x1_max)
    en1_x2 = np.append(en_x2, x2_max)

    # выбор точки x_j из плана
    new_pi = 1 / len(en1_x1)
    new_p = []
    for i in range(0, len(en1_x1)):
        new_p.append(pi)
    
    M = calc_M(en1_x1, en1_x2, new_p, m)
    D = calc_D(M)
    mind = 100
    x1_min = 2
    x2_min = 2
    i_min = 0
    for i in range(0, len(en1_x1)):
        d = calc_d(en1_x1[i], en1_x2[i], D)
        print("d = ", d)
        print("en1_x1[i] = ", en1_x1[i])
        print("en1_x2[i] = ", en1_x2[i])
        if mind >= d:
            mind = d
            x1_min = en1_x1[i]
            x2_min = en1_x2[i]
            i_min = i

    print("mind = ", mind)
    print("x1_min = ", x1_min)
    print("x2_min = ", x2_min)

    # точка x_j исключается из плана en1
    es1_x1 = np.delete(en1_x1, i_min)
    es1_x2 = np.delete(en1_x2, i_min)

    print("xs = ", x1_max, ", ", x2_max)
    print("xj = ", x1_min, ", ", x2_min)

    if (x1_max == x1_min) and (x2_max == x2_min):
        print("alg opt")
        print("|M(e)| = ", np.linalg.det(M), "ln|M(e)| = ", np.log(np.linalg.det(M)))
        print("es1_x1 = ", es1_x1)
        print("es1_x2 = ", es1_x2)
        draw_graph(es1_x1, es1_x2)
        break
    else:
        print("alg not opt")
        print("|M(e)| = ", np.linalg.det(M), "ln|M(e)| = ", np.log(np.linalg.det(M)))
        s += 1
        en_x1 = es1_x1
        en_x2 = es1_x2