import random
import math

# Method 1
def Selection1(n, pop, fitness):
    select = pop.copy()
    fitness1 = fitness.copy()
    Parents = []
    if sum(fitness1) == 0:
        for i in range(n):
            parent = select[random.randint(0,N-1)]
            Parents.append(parent)
    else:
        for i in range(4):
            arr = fitness1.index(min(fitness1))
            Parents.append(select[arr])
            del select[arr]
            del fitness1[arr]
            
    return Parents
# Method 2
def Selection2(n, pop_bin, fitness):
    select_bin = pop_bin.copy()
    fitness1 = fitness.copy()
    Parents = []
    if sum(fitness1) == 0:
        for i in range(n):
            parent = select_bin[random.randint(0,N-1)]
            Parents.append(parent)
    else: 
        NorParent = [(1 - indivi/sum(fitness1))/(N-1) for indivi in fitness1]
        tep = 0
        Cumulist = []
        for i in range(len(NorParent)):
            tep += NorParent[i]
            Cumulist.append(tep)
        #Find parents
        for i in range(n):
            z1 = random.uniform(0,1)
            for pick in range(len(Cumulist)):
                if z1<=Cumulist[0]:
                    parent = select_bin[NorParent.index(NorParent[0])]
                elif Cumulist[pick] < z1 <=Cumulist[pick+1]:
                    parent = select_bin[NorParent.index(NorParent[pick+1])]
            Parents.append(parent)
            
    return Parents
# 交配與突變
def Crossover_Mutation(parent1, parent2):
    def swap_machine(element_1, element_2):
        temp = element_1
        element_1 = element_2
        element_2 = temp
        return element_1, element_2
    child_1 = []
    child_2 = []
    for i in range(len(parent1)):
        z1 = random.uniform(0,1)
        if z1 < 0.9:
            z2 = random.uniform(0,1)
            cross_location = math.ceil(z2*(len(parent1[i])-1))
            #Crossover
            parent1[i][:cross_location],parent2[i][:cross_location] = swap_machine(parent1[i][:cross_location],parent2[i][:cross_location])
            p_list = [parent1[i], parent2[i]]
            for i in range(len(p_list)):
                z3 = random.uniform(0,1)
                if z3 < mr:
                    z4 = random.uniform(0,1)
                    temp_location = z4*(len(p_list[i])-1)
                    mutation_location = 0 if temp_location < 0.5 else math.ceil(temp_location)
                    p_list[i][mutation_location] = 0 if p_list[i][mutation_location] == 1 else 1
            child_1.append(p_list[0])
            child_2.append(p_list[1])
        else:
            child_1.append(parent1[i])
            child_2.append(parent2[i])
            
    return child_1,child_2