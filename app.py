# ============================= #
#          for ABC_v2           #
# ============================= #
import matplotlib.pyplot as plt
import numpy as np
from ypstruct import structure
import time
import ABC_v2

start = time.time()         #运行开始时刻
# 测试函数
def sphere(x):
    return sum(x**2)

# 问题定义
problem = structure()
problem.costfunc = sphere
problem.nvar = 10
problem.varmin = -100 * np.ones(10)
problem.varmax = 100 * np.ones(10)

# ABC参数
params = structure()
params.maxit = 500
params.npop = 100
params.nonlooker = 100
params.a = 1

# 运行ABC
out = ABC_v2.run(problem, params)
# 运行结果
plt.rcParams['font.sans-serif'] = ['KaiTi']  #设置字体为楷体
plt.plot(out.bestcost)
print("最优解：{}".format(out.bestsol))
end = time.time()              # 运行结束时刻
print('运行时间：{}s'.format(end-start))

plt.xlim(0, params.maxit)
plt.xlabel('迭代次数')
plt.ylabel('全局最优目标函数值')
plt.title('人工蜂群算法')
plt.grid(True)
plt.show()