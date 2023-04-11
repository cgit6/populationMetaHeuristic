import numpy as np
import random
import copy

def initialization(pop,ub,lb,dim):
    X = np.zeros([pop,dim]) #声明空间
    for i in range(pop):
        for j in range(dim):
            X[i,j]=(ub[j]-lb[j])*np.random.random()+lb[j] #生成[lb,ub]之间的随机数
    
    return X
     
def BorderCheck(X,ub,lb,pop,dim):
    for i in range(pop):
        for j in range(dim):
            if X[i,j]>ub[j]:
                X[i,j] = ub[j]
            elif X[i,j]<lb[j]:
                X[i,j] = lb[j]
    return X


def CaculateFitness(X,fun):
    pop = X.shape[0]
    fitness = np.zeros([pop, 1])
    for i in range(pop):
        fitness[i] = fun(X[i, :])
    return fitness


def SortFitness(Fit):
    fitness = np.sort(Fit, axis=0)
    index = np.argsort(Fit, axis=0)
    return fitness,index

def SortPosition(X,index):
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i,:] = X[index[i],:]
    return Xnew


def MFO(pop, dim, lb, ub, MaxIter, fun):
    r = 2; #参数
    X = initialization(pop,ub,lb,dim)  # 初始化种群
    fitness = CaculateFitness(X, fun)  # 计算适应度值
    fitnessS, sortIndex = SortFitness(fitness)  # 对适应度值排序
    Xs = SortPosition(X, sortIndex)  # 种群排序后，初始化火焰位置
    GbestScore = copy.copy(fitnessS[0]) #最优适应度值
    GbestPositon = np.zeros([1,dim])
    GbestPositon[0,:] = copy.copy(Xs[0,:])#最优解
    Curve = np.zeros([MaxIter, 1])
    for iter in range(MaxIter):
        Flame_no=round(pop-iter*((pop-1)/MaxIter)) #火焰数量更新
        r = -1 + iter*(-1)/MaxIter # r 线性从-1降到-2
        #飞蛾扑火行为
        for i in range(pop):
            for j in range(dim):
                if i<= Flame_no:
                    distance_to_flame = np.abs(Xs[i,j] - X[i,j]) #飞蛾与火焰的距离
                    b = 1
                    t = (r - 1)*random.random() + 1          
                    X[i,j] = distance_to_flame*np.exp(b*t)*np.cos(t*2*np.pi) + Xs[i,j] #螺旋飞行
                else:
                    distance_to_flame = np.abs(Xs[Flame_no,j] - X[i,j]) #飞蛾与火焰的距离
                    b = 1
                    t = (r - 1)*random.random() + 1
                    X[i,j] = distance_to_flame*np.exp(b*t)*np.cos(t*2*np.pi) + Xs[Flame_no,j] #螺旋飞行
        
        X = BorderCheck(X, ub, lb, pop, dim)  # 边界检测     
        fitness = CaculateFitness(X, fun)  # 计算适应度值
        fitnessS, sortIndex = SortFitness(fitness)  # 对适应度值排序
        Xs = SortPosition(X, sortIndex)  # 种群排序，作为下一代火焰的位置
        if fitnessS[0] <= GbestScore:  # 更新全局最优
            GbestScore = copy.copy(fitnessS[0])
            GbestPositon[0,:] = copy.copy(Xs[0, :])
        Curve[iter] = GbestScore
     
    return GbestScore, GbestPositon, Curve