import numpy as np
import copy as copy

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


def distance(a,b):
    d = np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
    return d



def S_func(r):
    f=0.5
    l=1.5
    o=f*np.exp(-r/l)-np.exp(-r)
    return o
        

def GOA(pop, dim, lb, ub, MaxIter, fun):
    #定义参数c的范围
    cMax = 1
    cMin = 0.00004
    X = initialization(pop,ub,lb,dim) #初始化种群
    fitness = CaculateFitness(X,fun) #计算适应度值
    fitness,sortIndex = SortFitness(fitness) #对适应度值排序
    X = SortPosition(X,sortIndex) #种群排序
    GbestScore = copy.copy(fitness[0])
    GbestPositon = copy.copy(X[0,:])
    Curve = np.zeros([MaxIter,1])
    GrassHopperPositions_temp = np.zeros([pop,dim])#应用于临时存放新位置
    for t in range(MaxIter):
        c = cMax - t*((cMax - cMin)/MaxIter) #计算参数c
        for i in range(pop):
            Temp = X.T
            S_i_total = np.zeros([dim,1])
            for k in range(0,dim-1,2):
                S_i = np.zeros([2,1])
                for j in range(pop):
                    if i != j:
                        Dist = distance(Temp[k:k+2,j],Temp[k:k+2,i])#计算两只蝗虫的距离d
                        r_ij_vec=(Temp[k:k+2,j]-Temp[k:k+2,i])/(Dist + 2**-52)#计算距离单位向量，2**-52是一个极小数，防止分母为0
                        xj_xi = 2 + Dist%2 #计算|xjd - xid|
                        s_ij1 = ((ub[k] - lb[k])*c/2)*S_func(xj_xi)*r_ij_vec[0]
                        s_ij2 = ((ub[k+1] - lb[k+1])*c/2)*S_func(xj_xi)*r_ij_vec[1]
                        S_i[0,:] = S_i[0,:] + s_ij1
                        S_i[1,:] = S_i[1,:] + s_ij2
                S_i_total[k:k+2,:]=S_i
            Xnew = c*S_i_total.T + GbestPositon #更新位置
            GrassHopperPositions_temp[i,:] = copy.copy(Xnew)
        
        X = BorderCheck(GrassHopperPositions_temp,ub,lb,pop,dim) #边界检测       
        fitness = CaculateFitness(X,fun) #计算适应度值
        fitness,sortIndex = SortFitness(fitness) #对适应度值排序
        X = SortPosition(X,sortIndex) #种群排序

        if(fitness[0]<=GbestScore): 
            GbestScore = copy.copy(fitness[0])
            GbestPositon = copy.copy(X[0,:])
        Curve[t] = GbestScore
    
    return GbestScore,GbestPositon,Curve
    # return GbestScore[0], GbestPositon[0], Curve