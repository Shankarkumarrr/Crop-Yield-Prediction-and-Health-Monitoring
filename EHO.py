import numpy as np
import random as rn
import time
import numpy.matlib

def HLBO(initsol, fname, xmin, xmax, Max_iter):
    pass


def PFA(initsol, fname, xmin, xmax, Max_iter):
    pass


class Proposed:
    pass

def CaculateClanCenter(numVar, numElephantInEachClan, Clan, cindex):
    ClanCenter = np.zeros((numVar))
    for Elephantindex in range(numElephantInEachClan):
        ClanCenter = ClanCenter + Clan[cindex, Elephantindex, :]
    ClanCenter = (1 / numElephantInEachClan) * ClanCenter
    return ClanCenter


def check_limit(Positions, ub, lb):  # Positions
    for i in range(Positions.shape[0]):
        Flag4ub = Positions[i, :] > ub
        Flag4lb = Positions[i, :] < lb
        Positions[i, :] = (Positions[i, :] * (~(Flag4ub + Flag4lb))) + ub * Flag4ub + lb * Flag4lb
    return Positions


def CombineClan(popsize, numClan, NewClan, fitness):
    j = 0
    popindex = 0
    Population = np.zeros((popsize, NewClan.shape[2]))
    costfit = np.zeros((popsize))
    while popindex < popsize:
        for clanindex in range(numClan):
            Population[popindex, :] = NewClan[clanindex, j, :]
            costfit[popindex] = fitness[clanindex, j]
            popindex = popindex + 1
        j = j + 1
    return Population, costfit


def EHO(Population, fobj, VRmin, VRmax, MaxFEs):
    MaxParValue = VRmax[0, :]
    MinParValue = VRmin[0, :]
    ct = time.time()
    fitness = fobj(Population)
    MinCost = np.min(fitness)
    ind = np.where(fitness == np.min(fitness))[0]
    [popsize, numVar] = np.shape(Population)
    numClan = 2
    numElephantInEachClan = popsize / numClan
    leaderpos = np.zeros((MaxFEs, numVar))
    leaderpos[0, :] = Population[ind, :]

    Keep = 1
    numElephantInEachClan = numElephantInEachClan * np.ones(numClan)
    dim = numVar
    alpha = 0.5
    beta = 0.1

    GenIndex = 0
    while GenIndex < MaxFEs:

        chromKeep = np.zeros((Keep, numVar))
        costKeep = np.zeros((popsize))
        for j in range(Keep):
            chromKeep[j, :] = Population[j, :]
            costKeep[j] = fitness[j]

        j = 0
        popindex = 0
        Clan = np.zeros((numClan, popsize, numVar))
        while popindex < popsize:
            for cindex in range(numClan):
                Clan[cindex, j, :] = Population[popindex, :]
                popindex = popindex + 1
            j = j + 1

        j = 0
        popindex = 0
        NewClan = np.zeros((numClan, popsize, numVar))
        while popindex <= popsize:
            for cindex in range(numClan):
                ClanCenter = CaculateClanCenter(numVar, np.round(numElephantInEachClan[cindex]).astype('int'), Clan,
                                                cindex)
                NewClan[cindex, j, :] = Clan[cindex, j, :] + alpha * (
                            Clan[cindex, 0, :] - Clan[cindex, j, :]) * np.random.random(dim)

                if sum(NewClan[cindex, j, :] - Clan[cindex, j, :]) == 0:
                    NewClan[cindex, j, :] = beta * ClanCenter
                popindex = popindex + 1

            j = j + 1
        for cindex in range(numClan):
            NewClan[cindex, -1:, :] = MinParValue + (MaxParValue - MinParValue + 1) * np.random.random(numVar)

        SavePopSize = popsize
        fitn = np.zeros((numClan, popsize))
        for i in range(numClan):
            NewClan[i] = check_limit(NewClan[i], MaxParValue, MinParValue)
            fitn[i, :] = fobj(NewClan[i])[:, 0]
        popsize = SavePopSize

        [Population, fitness] = CombineClan(popsize, numClan, NewClan, fitn)

        ind = np.where(fitness == np.min(fitness))[0]
        fitness = np.sort(fitness)

        Population = Population[ind, :]

        n = Population.shape[0]
        for k3 in range(Keep):
            Population[n - k3 - 1, :] = chromKeep[k3, :]
            fitness[n - k3 - 1] = costKeep[k3]

        ind = np.where(fitness == np.min(fitness))[0]
        fitness = np.sort(fitness)

        Population = Population[ind, :]

        MinCost = np.append(MinCost, fitness[0])

        leaderpos[GenIndex, :] = Population[0, :]
        GenIndex = GenIndex + 1
    bestmin = MinCost[:-1]
    ct = time.time() - ct

    return bestmin, MinCost, leaderpos, ct
