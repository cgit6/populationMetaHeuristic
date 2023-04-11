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


def BOA(pop, dim, lb, ub, MaxIter, fun):
    # 初始化种群
    X = initialization(pop,ub,lb,dim)  
    # 计算适应度值
    fitness = CaculateFitness(X, fun)  
    #寻找最优适应度位置
    indexBest = np.argmin(fitness) 
    #记录最优适应度值
    GbestScore = fitness[indexBest] 
    GbestPositon = np.zeros([1,dim])
    GbestPositon[0,:] = X[indexBest, :]
    X_new = copy.copy(X)
    Curve = np.zeros([MaxIter, 1])

    # 算法的參數
    #切换概率
    p=0.8 
    #功率指数a
    power_exponent=0.1  
    #感知形态c
    sensory_modality=0.1 

    # 迭代階段
    for t in range(MaxIter):      
        # 位置更新
        for i in range(pop):
            #刺激强度I的计算
            FP = sensory_modality*(fitness[i]**power_exponent) 
            #全局搜索
            if random.random() < p: 
                dis = random.random()*random.random()*GbestPositon - X[i,:]
                Temp = np.matrix(dis*FP)
                X_new[i,:] = X[i,:] + Temp[0,:]
            else:
                #局部搜索
                Temp = range(pop)
                #随机选择个体
                JK = random.sample(Temp,pop) 
                dis=random.random()*random.random()*X[JK[0],:]-X[JK[1],:]
                Temp = np.matrix(dis*FP)
                X_new[i,:] = X[i,:] + Temp[0,:]
            for j in range(dim):
                if X_new[i,j] > ub[j]:
                    X_new[i, j] = ub[j]
                if X_new[i,j] < lb[j]:
                    X_new[i, j] = lb[j]
            
            # 選擇機制
            #如果更优才更新
            if(fun(X_new[i,:])<fitness[i]):
                X[i,:] = copy.copy(X_new[i,:])
                fitness[i] = copy.copy(fun(X_new[i,:]))
        
        X = BorderCheck(X, ub, lb, pop, dim)  # 边界检测
        fitness = CaculateFitness(X, fun)  # 计算适应度值
        indexBest = np.argmin(fitness)
        if fitness[indexBest] <= GbestScore:  # 更新全局最优
            GbestScore = copy.copy(fitness[indexBest])
            GbestPositon[0,:] = copy.copy(X[indexBest, :])
        Curve[t] = GbestScore

    return GbestScore, GbestPositon, Curve
