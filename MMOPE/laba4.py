import numpy as np
import random
import matplotlib.pyplot as plt


def func(x, theta):
    return np.array([x[0] ** theta[1] * x[1] ** theta[2],
                     theta[0] * theta[1] * x[0] ** (theta[1] - 1) * x[1] ** theta[2],
                     theta[0] * theta[2] * x[0] ** theta[1] * x[1] ** (theta[2] - 1)])

def makeY(plan, theta, p = 0.2):
    Y = list(map(lambda x: theta[0] * x[0]**theta[1] * x[1]**theta[2], plan))
    Y = list(map(lambda y: y + random.normalvariate(0, p * y), Y))
    return Y

def makePlan(grid = np.linspace(0, 5, 1001), m = 10):
    return list(map(lambda x: [random.choice(grid), random.choice(grid)], range(m)))

def makeX(plan):
    return np.array(list(map(lambda x: [1.0, np.math.log(x[0]), np.math.log(x[1])], plan)))

def OLS(X, Y):
    thetahead = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)
    thetahead[0] = np.math.e ** thetahead[0]
    return thetahead

def makeM(x, N, theta):
    M = np.zeros((len(func(x[0], theta)), len(func(x[0], theta))))
    for i in range(len(x)):
        M += 1.0/N * make_partM(func(x[i], theta))
    return M

def make_partM(fx):
    M = np.zeros((len(fx), len(fx)))
    for i in range(len(fx)):
        for j in range(len(fx)):
            M[i][j] = fx[i] * fx[j]
    return M

def makeD(M):
    return np.linalg.inv(M)

def d(x, D, newx, thetahead):
    return np.dot(np.dot(func(x, thetahead), D), func(newx, thetahead).T)

def Delta(x, D, N, newx, thetahead):
    return 1./float(N) * (d(newx, D, newx, thetahead) - d(x, D, x, thetahead))\
           - 1./float(N)**2 * (d(x, D, x, thetahead) * d(newx, D, newx, thetahead) - d(x, D, newx, thetahead)**2)

def findMaxforOneX(x, D, N, grid, thetahead):
    maxdot = [grid[0], grid[0]]
    maxvalue = Delta(x, D, N, maxdot, thetahead)
    for x1 in grid:
        for x2 in grid:
            value = Delta(x, D, N, [x1, x2], thetahead)
            if value > maxvalue:
                maxvalue = value
                maxdot = [x1, x2]
    return [maxvalue, maxdot]

def findMaxforAll(X, D, N, grid, thetahead):
    listofmax = [findMaxforOneX(x,D,N,grid,thetahead) for x in X]
    return [*max(listofmax),listofmax.index(max(listofmax))]

def makeOptimalPlan(thetahead, plan, grid, N):
    eps = 0.001
    iteration = 0
    print(thetahead)
    while True:
        M = makeM(plan, N, thetahead)
        print("det", np.linalg.det(M))
        D = makeD(M)
        print(iteration)
        delta = findMaxforAll(plan, D, N, grid, thetahead)
        if delta[0] > eps:
            plan[delta[2]] = delta[1]
        else:
            break
        iteration += 1
    return plan

def RSS(Y, X, thetahead):
    Yhead = np.dot(X, thetahead)
    return np.dot(Y - Yhead, Y - Yhead)

def Experiment(theta, plan):
    ARSS = 0
    Ahead = 0
    for i in range(100):
        Y = makeY(plan, theta)
        Y = np.log(Y)
        X = makeX(plan)
        thetahead = OLS(X, Y)
        ARSS += RSS(Y, X, thetahead)
        Ahead += np.dot(thetahead - theta, thetahead - theta)
    ARSS /= 100
    Ahead /= 100
    return ARSS, Ahead

def draw_graph(points):
    for p in points:
        plt.scatter(p[0], p[1]) 
    plt.plot()
    plt.show()


N = 15
grid = np.linspace(0.1, 5, 50)
plan = makePlan(grid = grid)
theta = [0.4, 0.4, 0.4]
Y = makeY(plan, theta)
Y = np.log(Y)
X = makeX(plan)

thetahead = OLS(X, Y)
firstplan = makePlan(grid = grid, m = N)
optplan = makeOptimalPlan(thetahead, firstplan, grid, N)

ARSS, Ahead = Experiment(theta, optplan)
print("Optplan")
print(optplan)
print("Average RSS ", ARSS)
print("Average norm head ", Ahead)
draw_graph(optplan)

firstplan = makePlan(grid = grid, m = N)
ARSS, Ahead = Experiment(theta, firstplan)
print("Random plan")
print(firstplan)
print("Average RSS ", ARSS)
print("Average norm head ", Ahead)
draw_graph(firstplan)
