# -*- coding: utf-8 -*-

import numpy as np
seed=50
import random
random.seed(0)
import copy as copy

def initialization(pop, ub, lb, dim):
    ''' 种群初始化函数'''
    '''
    pop:为种群数量
    dim:每个个体的维度
    ub:每个维度的变量上边界，维度为[dim,1]
    lb:为每个维度的变量下边界，维度为[dim,1]
    X:为输出的种群，维度[pop,dim]
    '''
    X = np.zeros([pop, dim])  # 声明空间
    for i in range(pop):
        for j in range(dim):
            X[i, j] = (ub[j]-lb[j])*np.random.random()+lb[j]  # 生成[lb,ub]之间的随机数

    return X

# 邊界檢查  
def BorderCheck(X,ub,lb,pop,dim):
    for i in range(pop):
        for j in range(dim):
            if X[i,j]>ub[j]:
                X[i,j] = ub[j]
            elif X[i,j]<lb[j]:
                X[i,j] = lb[j]
    return X

# 計算適應值
def CaculateFitness(X,fun):

    pop = X.shape[0]
    fitness = np.zeros([pop, 1])
    for i in range(pop):
        fitness[i] = fun(X[i, :])
    return fitness

# 適應值排序
def SortFitness(Fit):
    fitness = np.sort(Fit, axis=0)
    index = np.argsort(Fit, axis=0)
    return fitness,index

# 針對適應值的順序對個體位置排序
def SortPosition(X,index):
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i,:] = X[index[i],:]
    return Xnew

# 輪盤賭策略
def RouletteWheelSelection(P):
    C = np.cumsum(P)#累加
    r = np.random.random()*C[-1]#定义选择阈值，将随机概率与总和的乘积作为阈值
    out = 0
    #若大于或等于阈值，则输出当前索引，并将其作为结果，循环结束
    for i in range(P.shape[0]):
        if r<C[i]:
            out = i
            break
    return out
        

# 人工蜂群算法
def ABC(pop, dim, lb, ub, MaxIter, fun):
    #limit 參數
    L = round(0.6*dim*pop)
    #計數器，用於與limit進行比較判定接下來的操作
    C = np.zeros([pop,1]) 
    #引领蜂数量
    nOnlooker=pop 
    # 初始化種群
    X= initialization(pop,ub,lb,dim)
    # 計算適應度值
    fitness = CaculateFitness(X,fun)
    # 對適應度值排序
    fitness, sortIndex = SortFitness(fitness) 
    # 種群排序
    X = SortPosition(X, sortIndex)  
    #記錄最優適應度值
    GbestScore = copy.copy(fitness[0])
    GbestPositon = np.zeros([1,dim])
    #記錄最優位置
    GbestPositon[0,:] = copy.copy(X[0, :])
    Curve = np.zeros([MaxIter, 1])
    Xnew = np.zeros([pop,dim])
    fitnessNew = copy.copy(fitness)
    # 迭代階段
    for t in range(MaxIter):
        '''引领蜂搜索'''
        for i in range(pop):
            #隨機選擇一個個體
            k = np.random.randint(pop)
            #當k=i時，再次隨機選擇，直到k不等於i
            while(k==i):
                k = np.random.randint(pop)
            phi = (2*np.random.random([1,dim]) - 1)
            #公式(2.2)位置更新
            Xnew[i,:] = X[i,:]+phi*(X[i,:]-X[k,:])
        #邊界檢查    
        Xnew=BorderCheck(Xnew,ub,lb,pop,dim)
        # 計算適應度值
        fitnessNew = CaculateFitness(Xnew, fun)  
        for i in range(pop):
            #如果適應度值更優，替換原始位置
            if fitnessNew[i]<fitness[i]:
                X[i,:]= copy.copy(Xnew[i,:])
                fitness[i] = copy.copy(fitnessNew[i])
            else:
                #如果位置沒有更新，累加器+1
                C[i] = C[i]+1 
                
        #計算選擇適應度權重
        F = np.zeros([pop,1])
        MeanCost = np.mean(fitness)
        for i in range(pop):
            F[i]=np.exp(-fitness[i]/MeanCost)
        #式（2.4）
        P=F/sum(F)

        '''侦察蜂搜索'''
        for m in range(nOnlooker):
            #輪盤賭測量選擇個體
            i=RouletteWheelSelection(P)
            #隨機選擇個體
            k = np.random.randint(pop)
            while(k==i):
                k = np.random.randint(pop)
            phi = (2*np.random.random([1,dim]) - 1)
            # 位置更新
            Xnew[i,:] = X[i,:]+phi*(X[i,:]-X[k,:])
            # 邊界檢查
        Xnew=BorderCheck(Xnew,ub,lb,pop,dim)
        # 計算適應值
        fitnessNew = CaculateFitness(Xnew,fun)  
        for i in range(pop):
            # 如果適應值更優，替換原始位置
            if fitnessNew[i]<fitness[i]:
                X[i,:]= copy.copy(Xnew[i,:])
                fitness[i] = copy.copy(fitnessNew[i])
            else:
                # 如果位置沒有更新，累加器+1
                C[i] = C[i]+1 
                
        '''判断limit条件，并进行更新'''
        for i in range(pop):
            if C[i]>=L:
                for j in range(dim):
                    X[i, j] = np.random.random() * (ub[j] - lb[j]) + lb[j]
                    C[i] = 0
        # 計算適應值
        fitness = CaculateFitness(X,fun)  
        # 對應適應度值排序
        fitness, sortIndex = SortFitness(fitness) 
        # 種群排序
        X = SortPosition(X, sortIndex)  
        # 更新全局最優
        if fitness[0] <= GbestScore: 
            GbestScore = copy.copy(fitness[0])
            GbestPositon[0,:] = copy.copy(X[0, :])
        Curve[t] = GbestScore

    return GbestScore, GbestPositon, Curve