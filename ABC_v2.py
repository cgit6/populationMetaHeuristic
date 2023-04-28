import numpy as np
from ypstruct import structure


def run(problem, params):
    # 函数信息
    costfunc = problem.costfunc
    nvar = problem.nvar
    varmin = problem.varmin
    varmax = problem.varmax

    # 参数信息
    maxit = params.maxit
    npop = params.npop
    nonlooker = params.nonlooker
    limit = int(np.round(0.6*nvar*npop))
    a = params.a

    # 空的蜂群结构
    empty_bee = structure()
    empty_bee.position = None
    empty_bee.cost = None

    # 临时蜂群结构
    newbee = structure()
    newbee.position = None
    newbee.cost = None

    # 初始化全局最优解
    bestsol = empty_bee.deepcopy()
    bestsol.cost = np.inf

    # 种群初始化
    pop = empty_bee.repeat(npop)

    for i in range(npop):
        pop[i].position = np.random.uniform(varmin, varmax, nvar)
        pop[i].cost = costfunc(pop[i].position)
        if pop[i].cost < bestsol.cost:
            bestsol = pop[i].deepcopy()

    # 初始化每个个体的抛弃次数
    count = np.empty(npop)

    # 记录每一代中全局最优个体目标函数值
    bestcost = np.empty(maxit)

    # 人工蜂群算法主循环
    for it in range(maxit):

        # 引领蜂
        for i in range(npop):

            # 随机选择k，不等于i
            K = np.append(np.arange(0, i), np.arange(i+1, npop))
            k = K[np.random.randint(K.size)]

            # 定义加速系数
            phi = a * np.random.uniform(-1, 1, nvar)

            # 新的蜜蜂位置
            newbee.position = pop[i].position + phi * \
                (pop[i].position - pop[k].position)

            # 计算新蜜蜂目标函数值
            newbee.cost = costfunc(newbee.position)

            # 通过比较目标函数值，更新第i个蜜蜂的位置
            if newbee.cost < pop[i].cost:
                pop[i] = newbee.deepcopy()
            else:
                count[i] += 1

        # 计算适应度值和选择概率
        fit = np.empty(npop)
        meancost = np.mean([pop[i].cost for i in range(npop)])
        for i in range(npop):
            fit[i] = np.exp(-pop[i].cost/meancost)  # 将目标函数值转换为适应度值

        probs = fit / np.sum(fit)

        # 跟随蜂
        for m in range(nonlooker):

            # 通过轮盘赌的方式选择蜜源
            i = roulette_wheel_selection(probs)

            # 随机选择k，不等于i
            K = np.append(np.arange(0, i), np.arange(i + 1, npop))
            k = K[np.random.randint(K.size)]

            # 定义加速系数
            phi = a * np.random.uniform(-1, 1, nvar)

            # 新的蜜蜂位置
            newbee.position = pop[i].position + phi * \
                (pop[i].position - pop[k].position)

            # 计算新蜜蜂目标函数值
            newbee.cost = costfunc(newbee.position)

            # 通过比较目标函数值，更新第i个蜜蜂的位置
            if newbee.cost < pop[i].cost:
                pop[i] = newbee.deepcopy()
            else:
                count[i] += 1

        # 侦察蜂
        for i in range(npop):
            if count[i] > limit:
                pop[i].position = np.random.uniform(varmin, varmax, nvar)
                pop[i].cost = costfunc(pop[i].position)
                count[i] = 0

        # 更新全局最优解
        for i in range(npop):
            if pop[i].cost < bestsol.cost:
                bestsol = pop[i].deepcopy()

        # 存储每一代全局最优解的目标函数值
        bestcost[it] = bestsol.cost

        # 展示迭代信息
        print("Iteration {}: Best Cost = {}".format(it, bestcost[it]))

    # 返回值
    out = structure()
    out.pop = pop
    out.bestsol = bestsol
    out.bestcost = bestcost
    return out


def roulette_wheel_selection(p):
    c = np.cumsum(p)
    r = sum(p) * np.random.rand()
    ind = np.argwhere(r <= c)
    return ind[0][0]
