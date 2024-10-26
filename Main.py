import cv2 as cv
import numpy as np
import os
import pandas as pd
import random as rn
from numpy import matlib
from DOX import DOX
from EHO import EHO, HLBO, PFA, Proposed
from Global_Vars import Global_Vars
from Model_3DCNN import Model_3DCNN
from Model_Adaboost import Model_Adaboost
from Model_Ensemble import Model_Ensemble
from Model_LogisticReg import Model_LogisticReg
from Model_SVM import Model_SVM
from Neural_Network import train_nn
from Objective_Function import objfun_feat, objfun_cls
from Plot_Results import plot_results_1, plot_results, plot_Convergence


def Read_Image(filename):
    image = cv.imread(filename)
    image = np.uint8(image)
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    image = cv.resize(image, (512, 512))
    return image


# Read Dataset  - Crop Images
an = 0
if an == 1:
    Directory = './Dataset/crop_images/'
    Out_Fold = os.listdir(Directory)
    Images = []
    Target = []
    for i in range(len(Out_Fold)):
        In_Fold = os.listdir(Directory + Out_Fold[i])
        for j in range(len(In_Fold)):
            print(i, j)
            filename = Directory + Out_Fold[i] + '/' + In_Fold[j]
            Image = Read_Image(filename)
            Target.append(i)
            Images.append(Image)
    np.save('Images.npy', Images)
    np.save('Target.npy', Target)

# Generate Target
an = 0
if an == 1:
    Target = np.load('Target.npy', allow_pickle=True)

    Target = np.asarray(Target)
    uniq = np.unique(Target)
    target = np.zeros((Target.shape[0], len(uniq)))
    for uni in range(len(uniq)):
        index = np.where(Target == uniq[uni])
        target[index[0], uni] = 1
    np.save('Targets.npy', target)

# First Set Feature - Deep Feature From Model 3DCNN
an = 0
if an == 1:
    Image = np.load('Images.npy', allow_pickle=True)
    Target = np.load('Targets.npy', allow_pickle=True)
    Feat = Model_3DCNN(Image, Target)
    np.save('CNN_Feature.npy', Feat)

# Second and Theird Feature - soil parameters and environmental data
an = 0
if an == 1:
    Data = pd.read_csv('Dataset/Crop_recommendation.csv')
    data = np.asarray(Data)
    Feat2 = data[:674, :3]
    Feat3 = data[:674, 3:7]
    Target = data[:674, 7]
    Target = np.asarray(Target)
    uniq = np.unique(Target)
    target = np.zeros((Target.shape[0], len(uniq)))
    for uni in range(len(uniq)):
        index = np.where(Target == uniq[uni])
        target[index[0], uni] = 1
    np.save('Feat2.npy', Feat2)
    np.save('Feat3.npy', Feat3)
    np.save('Targets2.npy', target)

# Feature Concatenation
an = 0
if an == 1:
    Feat1 = np.load('CNN_Feature.npy', allow_pickle=True)
    Feat2 = np.load('Feat2.npy', allow_pickle=True)
    Feat3 = np.load('Feat3.npy', allow_pickle=True)
    Feature = np.concatenate((Feat1, Feat2, Feat3), axis=1)  # Feature Concatenation step
    np.save('Feature.npy', Feature)  # Save Feature

# optimization for Feature Selection
an = 0
if an == 1:
    Feat = np.load('Feature.npy', allow_pickle=True)  # loading step
    Target = np.load('Targets.npy', allow_pickle=True)[:674, :]  # loading step
    Global_Vars.Feat = Feat
    Global_Vars.Target = Target
    Npop = 10
    Chlen = 25
    xmin = np.zeros((Npop, Chlen))
    xmax = np.ones((Npop, Chlen))
    fname = objfun_feat
    initsol = np.zeros((Npop, Chlen))
    for p1 in range(initsol.shape[0]):
        for p2 in range(initsol.shape[1]):
            initsol[p1, p2] = rn.uniform(xmin[p1, p2], xmax[p1, p2])
    Max_iter = 10

    print("EHO...")
    [bestfit1, fitness1, bestsol1, time] = EHO(initsol, fname, xmin, xmax, Max_iter)  # EHO

    print("DOX...")
    [bestfit2, fitness2, bestsol2, time] = DOX(initsol, fname, xmin, xmax, Max_iter)  # DOX

    print("HLBO...")
    [bestfit4, fitness4, bestsol4, time] = HLBO(initsol, fname, xmin, xmax, Max_iter)  # HLBO

    print("PFA...")
    [bestfit3, fitness3, bestsol3, time] = PFA(initsol, fname, xmin, xmax, Max_iter)  # PFA

    print("Proposed...")
    [bestfit5, fitness5, bestsol5, time] = Proposed(initsol, fname, xmin, xmax, Max_iter)  # Proposed
    BestSol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]   # Bestsol

    np.save('BestSol_Feat.npy', BestSol)  # Save Bestsol

# Optimized Feature Selection
an = 0
if an == 1:
    Feat = np.load('Feature.npy', allow_pickle=True)  # loading step
    Target = np.load('Targets.npy', allow_pickle=True)  # loading step
    BestSol = np.load('BestSol_Feat.npy', allow_pickle=True)  # loading step
    Feature = []
    for i in range(len(BestSol)):
        sol = np.round(BestSol[i, :]).astype(np.int16)
        feat = Feat[:, sol]
        Feature.append(feat)
    np.save('Selected_Features.npy', Feature)  # Save Selected Feature

# Optimization for crop yield prediction
an = 0
if an == 1:
    Feat = np.load('Selected_Features.npy', allow_pickle=True)
    Target = np.load('Targets.npy', allow_pickle=True)
    Global_Vars.Feat = Feat
    Global_Vars.Target = Target
    Npop = 10
    Chlen = 5  # SVM -> Kernal, ANN -> Hidden Neuron count, Learning Rate, Logistic Regression, Adaboost -> no of
    # estimators, Learning Rate
    xmin = matlib.repmat([0, 5, 0.01, 5, 0.01], Npop, 1)
    xmax = matlib.repmat([3, 255, 0.99, 50, 0.99], Npop, 1)
    fname = objfun_cls
    initsol = np.zeros((Npop, Chlen))
    for p1 in range(initsol.shape[0]):
        for p2 in range(initsol.shape[1]):
            initsol[p1, p2] = rn.uniform(xmin[p1, p2], xmax[p1, p2])
    Max_iter = 10

    print("EHO...")
    [bestfit1, fitness1, bestsol1, time] = EHO(initsol, fname, xmin, xmax, Max_iter)  # EHO

    print("DOX...")
    [bestfit2, fitness2, bestsol2, time] = DOX(initsol, fname, xmin, xmax, Max_iter)  # DOX

    print("HLBO...")
    [bestfit4, fitness4, bestsol4, time] = HLBO(initsol, fname, xmin, xmax, Max_iter)  # HLBO

    print("PFA...")
    [bestfit3, fitness3, bestsol3, time] = PFA(initsol, fname, xmin, xmax, Max_iter)  # PFA

    print("Proposed...")
    [bestfit5, fitness5, bestsol5, time] = Proposed(initsol, fname, xmin, xmax, Max_iter)  # Proposed

    BestSol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]   # Bestsol

    np.save('BestSol_CLS.npy', BestSol)

# Classification for Crop yield predicted and health monitored output
an = 0
if an == 1:
    Feature = np.load('Selected_Features.npy', allow_pickle=True)
    Target = np.load('Targets.npy', allow_pickle=True)
    BestSol = np.load('BestSol_CLS.npy', allow_pickle=True)
    Feat = Feature
    EVAL = []
    Learnper = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
    for learn in range(len(Learnper)):
        learnperc = round(Feat.shape[0] * Learnper[learn])
        Train_Data = Feat[:learnperc, :]
        Train_Target = Target[:learnperc, :]
        Test_Data = Feat[learnperc:, :]
        Test_Target = Target[learnperc:, :]
        Eval = np.zeros((10, 14))
        sol = np.round(BestSol).astype(np.int16)
        Eval[5, :], pred = Model_SVM(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[6, :], pred = train_nn(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[7, :], pred = Model_Adaboost(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[8, :], pred = Model_LogisticReg(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[9, :], pred = Model_Ensemble(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[9, :], pred = Model_Ensemble(Train_Data, Train_Target, Test_Data, Test_Target, sol)
        EVAL.append(Eval)
    np.save('Eval_all.npy', EVAL)

if __name__ == '__main__':
    plot_results_1()
    plot_Convergence()
    plot_results()