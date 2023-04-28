# -*- coding: utf-8 -*-

import numpy as np
# np.random.seed(10)
import math

import random
# random.seed(0)

import copy as copy


def initialization(pop, ub, lb, dim):
    # 声明空间
    X = np.zeros([pop, dim])  
    for i in range(pop):
        for j in range(dim):
            # 生成[lb,ub]之间的随机数
            X[i, j] = (ub[j]-lb[j])*np.random.random()+lb[j]  

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

# =========================================================== #
#                            ABC                              #
# =========================================================== #
# 輪盤賭策略
def RouletteWheelSelection(P):
    # 概率累加
    C = np.cumsum(P)
    # 定義選擇閾值，將隨機概率與總和的乘積作為閾值
    r = np.random.random()*C[-1]
    out = 0
    # 若大於或等於閾值，則輸出當前索引，並將其作為結果，循環結束
    for i in range(P.shape[0]):
        if r<C[i]:
            out = i
            break
    return out


def ABC(pop, dim, lb, ub, MaxIter, fun):
    # limit 參數
    L = round(0.6*dim*pop)
    # 計數器，用於與limit進行比較判定接下來的操作
    C = np.zeros([pop,1]) 
    # 引领蜂数量
    nOnlooker=pop 
    # 初始化種群
    X= initialization(pop,ub,lb,dim)
    # 計算適應度值
    fitness = CaculateFitness(X,fun)
    # 對適應度值排序
    fitness, sortIndex = SortFitness(fitness) 
    # 種群排序
    X = SortPosition(X, sortIndex)  
    # 記錄最優適應度值
    GbestScore = copy.copy(fitness[0])
    GbestPositon = np.zeros([1,dim])
    # 記錄最優位置
    GbestPositon[0,:] = copy.copy(X[0, :])
    Curve = np.zeros([MaxIter, 1])
    Xnew = np.zeros([pop,dim])
    fitnessNew = copy.copy(fitness)
    # 迭代階段
    for t in range(MaxIter):
        for i in range(pop):
            # 隨機選擇一個個體
            k = np.random.randint(pop)
            # 當k=i時，再次隨機選擇，直到k不等於i
            while(k==i):
                k = np.random.randint(pop)
            phi = (2*np.random.random([1,dim]) - 1)
            # 公式(2.2)位置更新
            Xnew[i,:] = X[i,:]+phi*(X[i,:]-X[k,:])
        # 邊界檢查    
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

# =========================================================== #
#                            BOA                              #
# =========================================================== #
def BOA(pop, dim, lb, ub, MaxIter, fun):
    # 初始化种群
    X = initialization(pop, ub, lb, dim)
    # 计算适应度值
    fitness = CaculateFitness(X, fun)
    #寻找最优适应度位置
    indexBest = np.argmin(fitness)
    #记录最优适应度值
    GbestScore = fitness[indexBest]
    GbestPositon = np.zeros([1, dim])
    GbestPositon[0, :] = X[indexBest, :]
    X_new = copy.copy(X)
    Curve = np.zeros([MaxIter, 1])

    # 算法的參數
    #切换概率
    p = 0.8
    #功率指数a
    power_exponent = 0.1
    #感知形态c
    sensory_modality = 0.1

    # 迭代階段
    for t in range(MaxIter):
        # 位置更新
        for i in range(pop):
            #刺激强度I的计算
            FP = sensory_modality*(fitness[i]**power_exponent)
            #全局搜索
            if random.random() < p:
                dis = random.random()*random.random()*GbestPositon - X[i, :]
                Temp = np.matrix(dis*FP)
                X_new[i, :] = X[i, :] + Temp[0, :]
            else:
                #局部搜索
                Temp = range(pop)
                #随机选择个体
                JK = random.sample(Temp, pop)
                dis = random.random()*random.random()*X[JK[0], :]-X[JK[1], :]
                Temp = np.matrix(dis*FP)
                X_new[i, :] = X[i, :] + Temp[0, :]
            for j in range(dim):
                if X_new[i, j] > ub[j]:
                    X_new[i, j] = ub[j]
                if X_new[i, j] < lb[j]:
                    X_new[i, j] = lb[j]

            # 選擇機制
            #如果更优才更新
            if(fun(X_new[i, :]) < fitness[i]):
                X[i, :] = copy.copy(X_new[i, :])
                fitness[i] = copy.copy(fun(X_new[i, :]))

        X = BorderCheck(X, ub, lb, pop, dim)  # 边界检测
        fitness = CaculateFitness(X, fun)  # 计算适应度值
        indexBest = np.argmin(fitness)
        if fitness[indexBest] <= GbestScore:  # 更新全局最优
            GbestScore = copy.copy(fitness[indexBest])
            GbestPositon[0, :] = copy.copy(X[indexBest, :])
        Curve[t] = GbestScore

    return GbestScore, GbestPositon, Curve

# =========================================================== #
#                            CSA                              #
# =========================================================== #
def CSA(pop,dim,lb,ub,MaxIter,fun):
    
    #初始化种群
    X = initialization(pop,ub,lb,dim) 
    #计算适应度值
    fitness = CaculateFitness(X,fun) 
    #对适应度值排序
    fitness,sortIndex = SortFitness(fitness) 
    #种群排序
    X = SortPosition(X,sortIndex) 
    GbestScore = copy.copy(fitness[0])
    GbestPositon = copy.copy(X[0,:])
    Curve = np.zeros([MaxIter,1])
    
    
    for t in range(MaxIter):
        # 更新後的位置
        X_new = copy.copy(X)
        # 迭代階段
        for t in range(MaxIter):
            for i in range(pop):
                a = 1
                r1= a * (1 - (t / MaxIter))
                r2 = 180*np.random.random(dim)
                r3 = np.random.random(dim)
                r4 = np.random.random()
                # 位置更新
                # r4 判斷
                if r4 < 0.5 :
                    # 執行 sine 公式位置更新
                    # 新位置 = 當前位置 + r1 * sine * r2 * | r3 * 當前最優解 - 當前位置| 
                    Temp = r3 * GbestPositon - X[i, :]
                    X_new[i, :] = X[i, :] + r1 * np.sin(r2) * abs(Temp)
                else:
                    # 執行 cosine 公式位置更新
                    Temp = r3 * GbestPositon - X[i, :]
                    X_new[i, :] = X[i, :] + r1 * np.cos(r2) * abs(Temp)

                # 邊界調整
                for j in range(dim):
                    # 如果超過上界
                    if X_new[i,j] > ub[j]:
                        X_new[i, j] = ub[j]
                    # 如果超過下界
                    if X_new[i,j] < lb[j]:
                        X_new[i, j] = lb[j]
 
                # 選擇機制 : 如果更优才更新
                if(fun(X_new[i,:])<fitness[i]):
                    X[i,:] = copy.copy(X_new[i,:])
                    fitness[i] = copy.copy(fun(X_new[i,:]))
    
        #边界检测
        X = BorderCheck(X, ub, lb, pop, dim)
        #计算适应度值
        fitness = CaculateFitness(X, fun)
        #对适应度值排序
        fitness, sortIndex = SortFitness(fitness)
        #种群排序
        X = SortPosition(X, sortIndex)

        #更新全局最优
        if(fitness[0] <= GbestScore):
            GbestScore = copy.copy(fitness[0])
            GbestPositon = copy.copy(X[0, :])
        Curve[t] = GbestScore

    return GbestScore, GbestPositon, Curve
    # return GbestScore[0], GbestPositon[0][0], Curve

# =========================================================== #
#                            SMA                              #
# =========================================================== #
def SMA(pop,dim,lb,ub,MaxIter,fun):

    z = 0.03 #位置更新参数
    X = initialization(pop,ub,lb,dim) #初始化种群
    fitness = CaculateFitness(X,fun) #计算适应度值
    fitness,sortIndex = SortFitness(fitness) #对适应度值排序
    X = SortPosition(X,sortIndex) #种群排序
    GbestScore = copy.copy(fitness[0])
    GbestPositon = copy.copy(X[0,:])
    Curve = np.zeros([MaxIter,1])
    W = np.zeros([pop,dim]) #权重W矩阵
    for t in range(MaxIter):
        worstFitness = fitness[-1]
        bestFitness = fitness[0]
        S=bestFitness-worstFitness+ 10E-8 #当前最优适应度于最差适应度的差值，10E-8为极小值，避免分母为0；
        for i in range(pop):
            if i<pop/2: #适应度排前一半的W计算
                W[i,:]= 1+np.random.random([1,dim])*np.log10((bestFitness-fitness[i])/(S)+1)
            else:#适应度排后一半的W计算
                W[i,:]= 1-np.random.random([1,dim])*np.log10((bestFitness-fitness[i])/(S)+1)
        #惯性因子a,b
        tt = -(t/MaxIter)+1
        if tt!=-1 and tt!=1:
            a = np.math.atanh(tt)
        else:
            a = 1
        b = 1-t/MaxIter
        #位置更新
        for i in range(pop):
            if np.random.random()<z:
                #公式（1.4）第一个式子
                # 方法一
                X[i, :] = (ub.T-lb.T)*np.random.random([1, dim])+lb.T
                # 方法二
                # for k in range(dim):
                #     X[i, :] = (ub[k]-lb[k])*np.random.random([1, dim])+lb[k]
            else:
                p = np.tanh(abs(fitness[i]-GbestScore))
                vb = 2*a*np.random.random([1,dim])-a
                vc = 2*b*np.random.random([1,dim])-b
                for j in range(dim):
                    r = np.random.random()
                    A = np.random.randint(pop)
                    B = np.random.randint(pop)
                    if r<p:
                        #公式（1.4）第二个式子
                        X[i,j] = GbestPositon[j] + vb[0,j]*(W[i,j]*X[A,j]-X[B,j]) 
                        #公式(1.4)第三个式子
                        X[i,j] = vc[0,j]*X[i,j]         
        
        #边界检测 
        X = BorderCheck(X,ub,lb,pop,dim)
        #计算适应度值       
        fitness = CaculateFitness(X,fun) 
        #对适应度值排序
        fitness,sortIndex = SortFitness(fitness) 
        #种群排序
        X = SortPosition(X,sortIndex) 

        #更新全局最优
        if(fitness[0]<=GbestScore): 
            GbestScore = copy.copy(fitness[0])
            GbestPositon = copy.copy(X[0,:])
        Curve[t] = GbestScore
    
    return GbestScore,GbestPositon,Curve


# =========================================================== #
#                            ESMA
# =========================================================== #
def ESMA(pop,dim,lb,ub,MaxIter,fun):

    #初始化种群
    X = initialization(pop,ub,lb,dim) 
    #计算适应度值
    fitness = CaculateFitness(X,fun) 
    #对适应度值排序
    fitness,sortIndex = SortFitness(fitness) 
    #种群排序
    X = SortPosition(X,sortIndex) 
    GbestScore = copy.copy(fitness[0])
    GbestPositon = copy.copy(X[0,:])
    Curve = np.zeros([MaxIter,1])

    #算法参数
    z = 0.03 
    #权重W矩阵
    W = np.zeros([pop,dim]) 
    for t in range(MaxIter):
        worstFitness = fitness[-1]
        bestFitness = fitness[0]
        #当前最优适应度于最差适应度的差值，10E-8为极小值，避免分母为0；
        S=bestFitness-worstFitness+ 10E-8 

        # 計算權重
        for i in range(pop):

            #适应度排前一半的W计算
            if i<pop/2: 
                W[i,:]= 1+np.random.random([1,dim])*np.log10((bestFitness-fitness[i])/(S)+1)
            #适应度排后一半的W计算
            else:
                W[i,:]= 1-np.random.random([1,dim])*np.log10((bestFitness-fitness[i])/(S)+1)
        
        #惯性因子a,b
        k = 1-(t/MaxIter)
        if k!=-1 and k!=1:
            # sigmoid function
            a = (k+(1-1/(1+np.e**-0.5))*k**2)/(1+k)
        else:
            a = 1
        # for vc
        b = 1-t/MaxIter

        #位置更新
        for i in range(pop):
            if np.random.random()<z:
                #全局搜索
                X[i,:] = (ub.T-lb.T)*np.random.random([1,dim])+lb.T 
            else:

                # 局部搜索
                p = np.tanh(abs(fitness[i]-GbestScore))
                vb = 2*a*np.random.random([1,dim])-a
                vc = 2*b*np.random.random([1,dim])-2*b

                r1= b
                r2 = 180*np.random.random()
                r3 = np.random.random()
                r4 = np.random.random()

                for j in range(dim):
                    r = np.random.random()
                    A = np.random.randint(pop)
                    B = np.random.randint(pop)

                    if r<p:
                        temp = W[i,j]*X[A,j]-X[B,j]
                        if r3 < 0.5:
                            X[i,j] = GbestPositon[j] + r1*np.sin(r2)* vb[0,j]*(temp) 
                        else:
                            X[i,j] = GbestPositon[j] + r1*np.cos(r2)* vb[0,j]*(temp) 
                            
                    else: 
                        temp = vc[0,j] * X[i,j]
                        if r3 < 0.5:
                            X[i,j] = r1 * np.sin(r2)*temp
                        else:
                            X[i,j] = r1 * np.cos(r2)*temp

        #边界检测
        X = BorderCheck(X,ub,lb,pop,dim) 
        #计算适应度值
        fitness = CaculateFitness(X,fun) 
        #对适应度值排序
        fitness,sortIndex = SortFitness(fitness) 
        #种群排序
        X = SortPosition(X,sortIndex) 
        #更新全局最优
        if(fitness[0]<=GbestScore): 
            GbestScore = copy.copy(fitness[0])
            GbestPositon = copy.copy(X[0,:])
        Curve[t] = GbestScore
    
    return GbestScore,GbestPositon,Curve


def distance(a, b):
    d = np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
    return d


def S_func(r):
    f = 0.5
    l = 1.5
    o = f*np.exp(-r/l)-np.exp(-r)
    return o


# =========================================================== #
#                            GOA
# =========================================================== #
def GOA(pop, dim, lb, ub, MaxIter, fun):
    #定义参数c的范围
    cMax = 1
    cMin = 0.00004
    X = initialization(pop, ub, lb, dim)  # 初始化种群
    fitness = CaculateFitness(X, fun)  # 计算适应度值
    fitness, sortIndex = SortFitness(fitness)  # 对适应度值排序
    X = SortPosition(X, sortIndex)  # 种群排序
    GbestScore = copy.copy(fitness[0])
    GbestPositon = copy.copy(X[0, :])
    Curve = np.zeros([MaxIter, 1])
    GrassHopperPositions_temp = np.zeros([pop, dim])  # 应用于临时存放新位置
    for t in range(MaxIter):
        c = cMax - t*((cMax - cMin)/MaxIter)  # 计算参数c
        for i in range(pop):
            Temp = X.T
            S_i_total = np.zeros([dim, 1])
            for k in range(0, dim-1, 2):
                S_i = np.zeros([2, 1])
                for j in range(pop):
                    if i != j:
                        Dist = distance(
                            Temp[k:k+2, j], Temp[k:k+2, i])  # 计算两只蝗虫的距离d
                        # 计算距离单位向量，2**-52是一个极小数，防止分母为0
                        r_ij_vec = (Temp[k:k+2, j] -
                                    Temp[k:k+2, i])/(Dist + 2**-52)
                        xj_xi = 2 + Dist % 2  # 计算|xjd - xid|
                        s_ij1 = ((ub[k] - lb[k])*c/2)*S_func(xj_xi)*r_ij_vec[0]
                        s_ij2 = ((ub[k+1] - lb[k+1])*c/2) * \
                            S_func(xj_xi)*r_ij_vec[1]
                        S_i[0, :] = S_i[0, :] + s_ij1
                        S_i[1, :] = S_i[1, :] + s_ij2
                S_i_total[k:k+2, :] = S_i
            Xnew = c*S_i_total.T + GbestPositon  # 更新位置
            GrassHopperPositions_temp[i, :] = copy.copy(Xnew)

        X = BorderCheck(GrassHopperPositions_temp, ub, lb, pop, dim)  # 边界检测
        fitness = CaculateFitness(X, fun)  # 计算适应度值
        fitness, sortIndex = SortFitness(fitness)  # 对适应度值排序
        X = SortPosition(X, sortIndex)  # 种群排序

        if(fitness[0] <= GbestScore):
            GbestScore = copy.copy(fitness[0])
            GbestPositon = copy.copy(X[0, :])
        Curve[t] = GbestScore

    return GbestScore, GbestPositon, Curve
    # return GbestScore[0], GbestPositon[0], Curve


def GSA(pop,dim,lb,ub,MaxIter,fun):
    a = -math.pi
    b = math.pi
    gold = (np.sqrt(5)-1)/2 #黄金分割率
    x1 = a + (1 - gold)*(b-a) #黄金分割系数x1
    x2 = a + gold*(b-a) #黄金分割系数x2  
    X = initialization(pop,ub,lb,dim) #初始化种群
    fitness = CaculateFitness(X,fun) #计算适应度值
    fitness,sortIndex = SortFitness(fitness) #对适应度值排序
    X = SortPosition(X,sortIndex) #种群排序
    GbestScore = copy.copy(fitness[0])
    GbestPositon = np.zeros([1,dim])
    GbestPositon[0,:] = copy.copy(X[0,:])#记录最优置
    Curve = np.zeros([MaxIter,1])
    for t in range(MaxIter):
        #根据位置更新公式，更新位置
        for i in range(pop):
            r = np.random.random()
            r1 = 2*math.pi*r
            r2 = r*math.pi
            for j in range(dim):
                X[i,j] = X[i,j]*np.abs(np.sin(r1)) -r2*np.sin(r1)*np.abs(x1*GbestPositon[0,j] - x2*X[i,j])
        
        X = BorderCheck(X,ub,lb,pop,dim) #边界检测  
        fitness = CaculateFitness(X,fun) #计算适应度值
        #更新黄金分割系数
        for i in range(pop):
            if fitness[i]<GbestScore: #如果解优于当前最优解
                GbestScore = fitness[i]
                GbestPositon[0,:] = copy.copy(X[i,:])
                b = x2
                x2 = x1
                x1 = a +(1 - gold)*(b-a)
            else:
                a = x1
                x1 = x2
                x2 = a + gold*(b-a)
            
            if x1 == x2:#如果分割系数相同，随机重置分割系数
                a = -math.pi*np.random.random()
                b = math.pi*np.random.random()
                x1 = a+(1-gold)*(b-a)
                x2 = a+gold*(b-a)
            
        fitness,sortIndex = SortFitness(fitness) #对适应度值排序
        X = SortPosition(X,sortIndex) #种群排序
        if(fitness[0]<=GbestScore): #更新全局最优
            GbestScore = copy.copy(fitness[0])
            GbestPositon[0,:] = copy.copy(X[0,:])
        Curve[t] = GbestScore
    
    return GbestScore,GbestPositon,Curve
    # return GbestScore[0], GbestPositon[0][0], Curve


def MFO(pop, dim, lb, ub, MaxIter, fun):
    r = 2  # 参数
    X = initialization(pop, ub, lb, dim)  # 初始化种群
    fitness = CaculateFitness(X, fun)  # 计算适应度值
    fitnessS, sortIndex = SortFitness(fitness)  # 对适应度值排序
    Xs = SortPosition(X, sortIndex)  # 种群排序后，初始化火焰位置
    GbestScore = copy.copy(fitnessS[0])  # 最优适应度值
    GbestPositon = np.zeros([1, dim])
    GbestPositon[0, :] = copy.copy(Xs[0, :])  # 最优解
    Curve = np.zeros([MaxIter, 1])
    for iter in range(MaxIter):
        Flame_no = round(pop-iter*((pop-1)/MaxIter))  # 火焰数量更新
        r = -1 + iter*(-1)/MaxIter  # r 线性从-1降到-2
        #飞蛾扑火行为
        for i in range(pop):
            for j in range(dim):
                if i <= Flame_no:
                    distance_to_flame = np.abs(Xs[i, j] - X[i, j])  # 飞蛾与火焰的距离
                    b = 1
                    t = (r - 1)*random.random() + 1
                    X[i, j] = distance_to_flame * \
                        np.exp(b*t)*np.cos(t*2*np.pi) + Xs[i, j]  # 螺旋飞行
                else:
                    distance_to_flame = np.abs(
                        Xs[Flame_no, j] - X[i, j])  # 飞蛾与火焰的距离
                    b = 1
                    t = (r - 1)*random.random() + 1
                    X[i, j] = distance_to_flame * \
                        np.exp(b*t)*np.cos(t*2*np.pi) + Xs[Flame_no, j]  # 螺旋飞行
        # 边界检测
        X = BorderCheck(X, ub, lb, pop, dim) 
        # 计算适应度值
        fitness = CaculateFitness(X, fun)  
        # 对适应度值排序
        fitnessS, sortIndex = SortFitness(fitness)  
        # 种群排序，作为下一代火焰的位置
        Xs = SortPosition(X, sortIndex)  
        if fitnessS[0] <= GbestScore:  # 更新全局最优
            GbestScore = copy.copy(fitnessS[0])
            GbestPositon[0, :] = copy.copy(Xs[0, :])
        Curve[iter] = GbestScore

    return GbestScore, GbestPositon, Curve
    return GbestScore[0], GbestPositon[0][0], Curve


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


def SSA(pop,dim,lb,ub,Max_iter,fun):
    ST = 0.8 #预警值
    PD = 0.2 #发现者的比列，剩下的是加入者
    SD = 0.1 #意识到有危险麻雀的比重
    PDNumber = int(pop*PD) #发现者数量
    SDNumber = int(pop*SD) #意识到有危险麻雀数量
    X = initialization(pop,ub,lb,dim) #初始化种群
    fitness = CaculateFitness(X,fun) #计算适应度值
    fitness,sortIndex = SortFitness(fitness) #对适应度值排序
    X = SortPosition(X,sortIndex) #种群排序
    GbestScore = copy.copy(fitness[0])
    GbestPositon = np.zeros([1,dim])
    GbestPositon[0,:] = copy.copy(X[0,:])
    Curve = np.zeros([Max_iter,1])
    for t in range(Max_iter):
        BestF = copy.copy(fitness[0])
        Xworst = copy.copy(X[-1,:])
        Xbest = copy.copy(X[0,:])
        R2 = np.random.random()
        for i in range(PDNumber):
            if R2<ST:
                X[i,:] = X[i,:]*np.exp(-i/(np.random.random()*Max_iter))
            else:
                X[i,:] = X[i,:] + np.random.randn()*np.ones([1,dim])
        
        X = BorderCheck(X,ub,lb,pop,dim) #边界检测   
        fitness = CaculateFitness(X,fun) #计算适应度值
        
        bestII=np.argmin(fitness)
        Xbest = copy.copy(X[bestII,:])
        for i in range(PDNumber+1,pop):
             if i>(pop - PDNumber)/2 + PDNumber:
                 X[i,:]= np.random.randn()*np.exp((Xworst - X[i,:])/i**2)
             else:
                 #产生-1，1的随机数
                 A = np.ones([dim,1])
                 for a in range(dim):
                     if(np.random.random()>0.5):
                         A[a]=-1       
                 AA = np.dot(A,np.linalg.inv(np.dot(A.T,A)))
                 X[i,:]= X[0,:] + np.abs(X[i,:] - GbestPositon)*AA.T
        
        X = BorderCheck(X,ub,lb,pop,dim) #边界检测   
        fitness = CaculateFitness(X,fun) #计算适应度值
        Temp = range(pop)
        RandIndex = random.sample(Temp, pop) 
        SDchooseIndex = RandIndex[0:SDNumber]#随机选取对应比列的麻雀作为意识到危险的麻雀
        for i in range(SDNumber):
            if fitness[SDchooseIndex[i]]>BestF:
                X[SDchooseIndex[i],:] = Xbest + np.random.randn()*np.abs(X[SDchooseIndex[i],:] - Xbest)
            elif fitness[SDchooseIndex[i]] == BestF:
                K = 2*np.random.random() - 1
                X[SDchooseIndex[i],:] = X[SDchooseIndex[i],:] + K*(np.abs( X[SDchooseIndex[i],:] - X[-1,:])/(fitness[SDchooseIndex[i]] - fitness[-1] + 10E-8))
        
        X = BorderCheck(X,ub,lb,pop,dim) #边界检测   
        fitness = CaculateFitness(X,fun) #计算适应度值
        fitness,sortIndex = SortFitness(fitness) #对适应度值排序
        X = SortPosition(X,sortIndex) #种群排序
        if(fitness[0]<GbestScore): #更新全局最优
            GbestScore = copy.copy(fitness[0])
            GbestPositon[0,:] = copy.copy(X[0,:])
        Curve[t] = GbestScore
    
    return GbestScore,GbestPositon,Curve
    # return GbestScore[0], GbestPositon[0][0], Curve


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


def WOA(pop, dim, lb, ub, MaxIter, fun):
    X = initialization(pop, ub, lb, dim)  # 初始化种群
    fitness = CaculateFitness(X, fun)  # 计算适应度值
    fitness, sortIndex = SortFitness(fitness)  # 对适应度值排序
    X = SortPosition(X, sortIndex)  # 种群排序
    GbestScore = copy.copy(fitness[0])  # 记录最优适应度值
    GbestPositon = np.zeros([1, dim])
    GbestPositon[0, :] = copy.copy(X[0, :])  # 记录最优解
    Curve = np.zeros([MaxIter, 1])
    for t in range(MaxIter):
        Leader = X[0, :]  # 领头鲸鱼
        a = 2 - t * (2 / MaxIter)  # 线性下降权重2 - 0
        for i in range(pop):
            r1 = random.random()
            r2 = random.random()

            A = 2 * a * r1 - a
            C = 2 * r2
            b = 1
            l = 2 * random.random() - 1  # [-1,1]之间的随机数

            for j in range(dim):
                p = random.random()
                if p < 0.5:
                    if np.abs(A) >= 1:  # 寻找猎物
                        rand_leader_index = min(
                            int(np.floor(pop * random.random() + 1)), pop - 1)  # 随机选择一个个体
                        X_rand = X[rand_leader_index, :]
                        D_X_rand = np.abs(C * X_rand[j] - X[i, j])
                        X[i, j] = X_rand[j] - A * D_X_rand
                    elif np.abs(A) < 1:  # 包围猎物
                        D_Leader = np.abs(C * Leader[j] - X[i, j])
                        X[i, j] = Leader[j] - A * D_Leader
                elif p >= 0.5:  # 气泡网攻击
                    distance2Leader = np.abs(Leader[j] - X[i, j])
                    X[i, j] = distance2Leader * \
                        np.exp(b * l) * np.cos(l * 2 * math.pi) + Leader[j]

        X = BorderCheck(X, ub, lb, pop, dim)  # 边界检测
        fitness = CaculateFitness(X, fun)  # 计算适应度值
        fitness, sortIndex = SortFitness(fitness)  # 对适应度值排序
        X = SortPosition(X, sortIndex)  # 种群排序
        if fitness[0] <= GbestScore:  # 更新全局最优
            GbestScore = copy.copy(fitness[0])
            GbestPositon[0, :] = copy.copy(X[0, :])
        Curve[t] = GbestScore

    return GbestScore, GbestPositon, Curve
    # return GbestScore[0], GbestPositon[0][0], Curve
