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


def TLBO(pop, dim, lb, ub, MaxIter, fun):
    X = initialization(pop,ub,lb,dim)  # 初始化种群
    fitness = CaculateFitness(X, fun)  # 计算适应度值
    GbestScore = np.min(fitness) #寻找最优适应度值
    indexBest = np.argmin(fitness) #最优适应度值对应得索引
    GbestPositon = np.zeros([1,dim])
    GbestPositon[0,:] = copy.copy(X[indexBest, :])#记录最优解
    Curve = np.zeros([MaxIter, 1])
    for t in range(MaxIter):
        for i in range(pop):
            #教阶段
            Xmean = np.mean(X) #计算平均位置
            indexBest = np.argmin(fitness) #寻找最优位置      
            Xteacher = copy.copy(X[indexBest,:]) #老师的位置，即最优位置
            beta = random.randint(0,1)#教学因子
            Xnew = X[i,:] + np.random.random(dim)*(Xteacher - beta*Xmean) #教阶段位置更新
            #边界检查
            for j in range(dim):
                if Xnew[j]>ub[j]:
                    Xnew[j] = ub[j]
                if Xnew[j]<lb[j]:
                    Xnew[j]=lb[j]      
            #计算新位置适应度
            fitnessNew = fun(Xnew);
            #如果新位置更优，则更新先前解
            if fitnessNew<fitness[i]:
                X[i,:] = copy.copy(Xnew)
                fitness[i] = copy.copy(fitnessNew)
            #学阶段
            p = random.randint(0,dim-1)#随机选择一个索引
            while i == p:#确保随机选择的索引不等于当前索引
                p = random.randint(0,dim-1)
            #学阶段位置更新
            if fitness[i]<fitness[p]:
                Xnew = X[i,:] + np.random.random(dim)*(X[i,:] - X[p,:])
            else:
                Xnew = X[i,:] - np.random.random(dim)*(X[i,:] - X[p,:])
            #边界检查
            for j in range(dim):
                if Xnew[j]>ub[j]:
                    Xnew[j] = ub[j]
                if Xnew[j]<lb[j]:
                    Xnew[j]=lb[j]
            #如果新位置更优，则更新先前解
            fitnessNew = fun(Xnew)
            #如果新位置更优，则更新先前解
            if fitnessNew<fitness[i]:
                X[i,:] = copy.copy(Xnew)
                fitness[i] = fitnessNew
                             
        fitness = CaculateFitness(X, fun)  # 计算适应度值
        indexBest = np.argmin(fitness)
        if fitness[indexBest] <= GbestScore:  # 更新全局最优
            GbestScore = copy.copy(fitness[indexBest])
            GbestPositon[0,:] = copy.copy(X[indexBest, :])
        Curve[t] = GbestScore

    return GbestScore, GbestPositon, Curve
    # return GbestScore[0], GbestPositon[0][0], Curve

