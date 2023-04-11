import numpy as np
import random
import copy


def initialization(pop, ub, lb, dim):
    X = np.zeros([pop, dim])  # 声明空间
    for i in range(pop):
        for j in range(dim):
            X[i, j] = (ub[j]-lb[j])*np.random.random()+lb[j]  # 生成[lb,ub]之间的随机数

    return X


def BorderCheck(X, ub, lb, pop, dim):
    for i in range(pop):
        for j in range(dim):
            if X[i, j] > ub[j]:
                X[i, j] = ub[j]
            elif X[i, j] < lb[j]:
                X[i, j] = lb[j]
    return X


def CaculateFitness(X, fun):
    pop = X.shape[0]
    fitness = np.zeros([pop, 1])
    for i in range(pop):
        fitness[i] = fun(X[i, :])
    return fitness


def SortFitness(Fit):
    fitness = np.sort(Fit, axis=0)
    index = np.argsort(Fit, axis=0)
    return fitness, index


def SortPosition(X, index):
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i, :] = X[index[i], :]
    return Xnew


def SOA(pop, dim, lb, ub, MaxIter, fun):
    fc = 2  # 可调
    X = initialization(pop, ub, lb, dim)  # 初始化种群
    fitness = CaculateFitness(X, fun)  # 计算适应度值
    fitness, sortIndex = SortFitness(fitness)  # 对适应度值排序
    X = SortPosition(X, sortIndex)  # 种群排序
    GbestScore = copy.copy(fitness[0])
    GbestPositon = np.zeros([1, dim])
    GbestPositon[0, :] = copy.copy(X[0, :])
    Curve = np.zeros([MaxIter, 1])
    MS = np.zeros([pop, dim])
    CS = np.zeros([pop, dim])
    DS = np.zeros([pop, dim])
    X_new = copy.copy(X)
    for i in range(MaxIter):
        Pbest = X[0, :]
        for j in range(pop):
            # 计算Cs
            A = fc - (i*(fc/MaxIter))
            CS[j, :] = X[j, :]*A
            # 计算Ms
            rd = np.random.random()
            B = 2*(A**2)*rd
            MS[j, :] = B*(Pbest - X[j, :])
            # 计算Ds
            DS[j, :] = np.abs(CS[j, :] + MS[j, :])
            # 局部搜索
            u = 1
            v = 1
            theta = np.random.random()
            r = u*np.exp(theta*v)
            x = r*np.cos(theta*2*np.pi)
            y = r*np.sin(theta*2*np.pi)
            z = r*theta
            # 攻击
            X_new[j, :] = x*y*z*DS[j, :] + Pbest

        X = BorderCheck(X_new, ub, lb, pop, dim)  # 边界检测
        fitness = CaculateFitness(X, fun)  # 计算适应度值
        fitness, sortIndex = SortFitness(fitness)  # 对适应度值排序
        X = SortPosition(X, sortIndex)  # 种群排序
        if(fitness[0] <= GbestScore):  # 更新全局最优
            GbestScore = copy.copy(fitness[0])
            GbestPositon[0, :] = copy.copy(X[0, :])
        Curve[i] = GbestScore

    return GbestScore, GbestPositon, Curve
    # return GbestScore[0], GbestPositon[0][0], Curve
