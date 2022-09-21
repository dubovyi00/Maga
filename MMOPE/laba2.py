import numpy as np

def function(x1, x2):
    return np.matrix([[1], [x1], [x2], [x1 * x2], [x1 ** 2], [x2 ** 2]])

def print_iteration(iter, M):
    print("iter = {}".format(iter) + "\ttrD = {:10.7f}".format(np.trace(np.linalg.inv(M))))

def print_plan(p):
    print("x1\tx2\tp")
    for i in range(n):
        if p[i] != 0:
            print("{:4}".format(x[i][0]) + "\t{:4}".format(x[i][1]) + "\t{:9.7f}".format(p[i]))
    print()

def optimal_plan(x, p):
    proj, grad = np.zeros(n), np.zeros(n)
    solution, iter = 0, 1
    eps = 1e-07
    while solution == 0:
        λ = 0.1
        solution = 1
        M = np.zeros((6, 6)) # Информационная матрица Фишера
        for i in range(0, n):
            M += p[i] * function(x[i][0], x[i][1]) *np.transpose(function(x[i][0], x[i][1]))
        M2 = np.linalg.matrix_power(M, -2)
        for i in range(0, n):
            grad[i] = np.trace(M2 * function(x[i][0], x[i][1]) *np.transpose(function(x[i][0], x[i][1])))
        grad /= np.linalg.norm(grad)
        avg = 0.0
        for i in range(0, n):
            if p[i] != 0.0:
                avg += grad[i]
        avg /= np.count_nonzero(p) # Количество ненулевых элементов весов p
        for i in range(0, n):
            if p[i] != 0 and abs(grad[i] - avg) > eps:
                solution = 0
        for j in range(0, n):
            proj[j] = grad[j] - avg
            if p[j] == 0:
                if proj[j] > 0:
                    solution = 0
                else:
                    proj[j] = 0
        if iter % 500 == 0:
            print_iteration(iter, M)
            print_plan(p)
        if solution == 0:
            for i in range(0, n):
                if proj[i] < 0 and λ > - p[i] / proj[i]:
                    λ = - p[i] / proj[i]
            for i in range(0, n):
                p[i] += λ * proj[i]
        iter += 1
    print("\nРезультат:")
    print_iteration(iter, M)
    max = 0.0
    for i in range(n):
        tr = np.trace(M2 * (function(x[i][0], x[i][1]) * np.transpose(function(x[i][0], x[i][1]))))
        if tr > max:
            max = tr
    print("max:", '%.7f' % max)
    print("Оптимальный план:")
    print_plan(p)

x_t = [-1, -0.75, -0.25, 0, 0.25, 0.75, 1]
n = len(x_t) ** 2
# Начальный план
x = np.zeros((n, 2))
i = 0 
for x1 in x_t:
    for x2 in x_t:
        x[i][0] = x1
        x[i][1] = x2
        i += 1
print(x)
p = np.ones(n) / n
optimal_plan(x, p)
