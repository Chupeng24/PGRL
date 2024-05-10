import numpy as np

def das_dennis_recursion(ref_dirs, ref_dir, n_partitions, beta, depth):
    if depth == len(ref_dir) - 1:
        ref_dir[depth] = beta / (1.0 * n_partitions)
        ref_dirs.append(ref_dir[None, :])
    else:
        for i in range(beta + 1):
            ref_dir[depth] = 1.0 * i / (1.0 * n_partitions)
            das_dennis_recursion(ref_dirs, np.copy(ref_dir), n_partitions, beta - i, depth + 1)


def das_dennis(n_partitions, n_dim):
    if n_partitions == 0:
        return np.full((1, n_dim), 1 / n_dim)
    else:
        ref_dirs = []
        ref_dir = np.full(n_dim, np.nan)
        das_dennis_recursion(ref_dirs, ref_dir, n_partitions, n_partitions, 0)
        return np.concatenate(ref_dirs, axis=0)

class Pop():
    def __init__(self, fitness, pre):
        self.fitness = fitness
        self.pre = pre
        self.obj_dim = len(fitness)

def tri_get_pareto_set(Pop):
    S=[[] for i in range(len(Pop))]
    front = [[]]
    n=[0 for i in range(len(Pop))]
    rank = [0 for i in range(len(Pop))]

    for p in range(len(Pop)):
        S[p]=[]
        n[p]=0
        for q in range(len(Pop)):
            if Tri_Dominate(Pop[p],Pop[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif Tri_Dominate(Pop[q],Pop[p]):
                n[p] = n[p] + 1
        if n[p]==0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)
    i = 0
    while(front[i] != []):
        Q=[]
        for p in front[i]:
            for q in S[p]:
                n[q] =n[q] - 1
                if( n[q]==0):
                    rank[q]=i+1
                    if q not in Q:
                        Q.append(q)
        i = i+1
        front.append(Q)
    del front[len(front) - 1]
    NDSet=[]
    for Fi in front:
        NDSeti=[]
        for pi in Fi:
            NDSeti.append(Pop[pi])
        NDSet.append(NDSeti)
    return NDSet

def bi_get_pareto_set(Pop):
    S=[[] for i in range(len(Pop))]
    front = [[]]
    n=[0 for i in range(len(Pop))]
    rank = [0 for i in range(len(Pop))]

    for p in range(len(Pop)):
        S[p]=[]
        n[p]=0
        for q in range(len(Pop)):
            if Dominate(Pop[p],Pop[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif Dominate(Pop[q],Pop[p]):
                n[p] = n[p] + 1
        if n[p]==0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)
    i = 0
    while(front[i] != []):
        Q=[]
        for p in front[i]:
            for q in S[p]:
                n[q] =n[q] - 1
                if( n[q]==0):
                    rank[q]=i+1
                    if q not in Q:
                        Q.append(q)
        i = i+1
        front.append(Q)
    del front[len(front) - 1]
    NDSet=[]
    for Fi in front:
        NDSeti=[]
        for pi in Fi:
            NDSeti.append(Pop[pi])
        NDSet.append(NDSeti)
    return NDSet

# 3-objective
def Tri_Dominate(Pop1,Pop2):
    '''
    :param Pop1:
    :param Pop2:
    :return: If Pop1 dominate Pop2, return True
    '''
    if (Pop1.fitness[0]<Pop2.fitness[0] and Pop1.fitness[1]<Pop2.fitness[1] and Pop1.fitness[2]<Pop2.fitness[2]) or \
        (Pop1.fitness[0] <= Pop2.fitness[0] and Pop1.fitness[1] < Pop2.fitness[1] and Pop1.fitness[2]<Pop2.fitness[2]) or \
        (Pop1.fitness[0] < Pop2.fitness[0] and Pop1.fitness[1] <= Pop2.fitness[1] and Pop1.fitness[2]<Pop2.fitness[2]) or \
            (Pop1.fitness[0] < Pop2.fitness[0] and Pop1.fitness[1] <Pop2.fitness[1] and Pop1.fitness[2]<=Pop2.fitness[2]) or \
            (Pop1.fitness[0] <= Pop2.fitness[0] and Pop1.fitness[1] <= Pop2.fitness[1] and Pop1.fitness[2]<Pop2.fitness[2]) or \
            (Pop1.fitness[0] < Pop2.fitness[0] and Pop1.fitness[1] <= Pop2.fitness[1] and Pop1.fitness[2]<=Pop2.fitness[2]) or \
            (Pop1.fitness[0] <=Pop2.fitness[0] and Pop1.fitness[1] < Pop2.fitness[1] and Pop1.fitness[2]<=Pop2.fitness[2]):
        return True
    else:
        return False

# bi-objective
def Dominate(Pop1,Pop2):
    '''
    :param Pop1:
    :param Pop2:
    :return: If Pop1 dominate Pop2, return True
    '''
    if (Pop1.fitness[0]<Pop2.fitness[0] and Pop1.fitness[1]<Pop2.fitness[1]) or \
        (Pop1.fitness[0] <= Pop2.fitness[0] and Pop1.fitness[1] < Pop2.fitness[1]) or \
        (Pop1.fitness[0] < Pop2.fitness[0] and Pop1.fitness[1] <= Pop2.fitness[1]):
        return True
    else:
        return False