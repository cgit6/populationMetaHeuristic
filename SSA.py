import numpy as np
import copy
import random
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









