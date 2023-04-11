import numpy as np
import copy as copy

def initialization(pop,ub,lb,dim):
    #声明空间
    X = np.zeros([pop,dim]) 
    for i in range(pop):
        for j in range(dim):
            #生成[lb,ub]之间的随机数
            X[i,j] = (ub[j]-lb[j])*np.random.random() + lb[j] 
    
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







