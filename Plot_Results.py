import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable

def plot_results_1():
    # matplotlib.use('TkAgg')
    eval = np.load('Evaluate_all.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Recall', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Graph_Term = [0, 1, 2, 3]
    Algorithm = ['TERMS', 'EHO', 'DOX', 'HLBO', 'PFA', 'PROPOSED']
    Classifier = ['TERMS', 'SVM', 'ANN', 'ADABOOST', 'LOGISTIC REGRESSION', 'ENSEMBLE', 'PROPOSED']
    for i in range(eval.shape[0]):
        value = eval[i, 4, :, 4:]


    # for i in range(eval.shape[0]):
    for j in range(len(Graph_Term)):
        Graph = np.zeros((eval.shape[0], eval.shape[2]))
        for k in range(eval.shape[0]):
            for l in range(eval.shape[2]):
                if j == 9:
                    Graph[k, l] = eval[k, 4, l, Graph_Term[j] + 4]
                else:
                    Graph[k, l] = eval[k, 4, l, Graph_Term[j] + 4]
        Dataset = ['Concatenate\nFeature', 'Optimal\nFeature']
        X = np.arange(5)
        plt.plot(X, Graph[0, :5], color='r', linewidth=3, marker='x', markerfacecolor='b', markersize=16,
                 label="Concatenate Feature")
        plt.plot(X, Graph[1, :5], color='g', linewidth=3, marker='D', markerfacecolor='red', markersize=12,
                 label="Optimal Feature")
        # plt.plot(Dataset, Graph[:, 2], color='b', linewidth=3, marker='x', markerfacecolor='green', markersize=16,
        #          label="HLBO")
        # plt.plot(Dataset, Graph[:, 3], color='c', linewidth=3, marker='D', markerfacecolor='cyan', markersize=12,
        #          label="PFA")
        # plt.plot(Dataset, Graph[:, 4], color='k', linewidth=3, marker='x', markerfacecolor='black', markersize=16,
        #          label="PROPOSED")
        # plt.xlabel('No Of Datasets')
        plt.xticks(X + 0.10, ('EHO-OEL', 'DOX-OEL', 'HLBO-OEL', 'PFA-OEL', 'SPHLP-OEL'))
        plt.ylabel(Terms[Graph_Term[j]])
        # plt.ylim([80, 100])
        plt.legend(loc=4)
        path1 = "./Results/%s_line.png" % (Terms[Graph_Term[j]])
        plt.savefig(path1)
        plt.show()

        fig = plt.figure()
        # ax = plt.axes(projection="3d")
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        X = np.arange(6)
        ax.bar(X + 0.00, Graph[0, 5:], color='r', width=0.10, label="Concatenate Feature")
        ax.bar(X + 0.10, Graph[1, 5:], color='g', width=0.10, label="Optimal Feature")
        # ax.bar(X + 0.20, Graph[:, 7], color='b', width=0.10, label="ADABOOST")
        # ax.bar(X + 0.30, Graph[:, 8], color='m', width=0.10, label="LOGISTIC REGRESSION")
        # ax.bar(X + 0.40, Graph[:, 9], color='y', width=0.10, label="ENSEMBLE")
        # ax.bar(X + 0.50, Graph[:, 10], color='k', width=0.10, label="PROPOSED")
        plt.xticks(X + 0.10, ('SVM', 'ANN', 'ADABOOST', 'LOGISTIC\nREGRESSION', 'ENSEMBLE', 'SPHLP-OEL'))
        # plt.xlabel('No Of Datasets')
        plt.ylabel(Terms[Graph_Term[j]])
        plt.legend(loc=1)
        path1 = "./Results/%s_bar.png" % (Terms[Graph_Term[j]])
        plt.savefig(path1)
        plt.show()


def plot_results():
    # matplotlib.use('TkAgg')
    eval = np.load('Evaluate_all.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Graph_Term = [0, 8, 9]
    Algorithm = ['TERMS', 'EHO', 'DOX', 'HLBO', 'PFA', 'PROPOSED']
    Classifier = ['TERMS', 'SVM', 'ANN', 'ADABOOST', 'LOGISTIC REGRESSION', 'ENSEMBLE', 'PROPOSED']
    for i in range(1):
        value = eval[i, 4, :, 4:]

        # Table = PrettyTable()
        # Table.add_column(Algorithm[0], Terms)
        # for j in range(len(Algorithm) - 1):
        #     Table.add_column(Algorithm[j + 1], value[j, :])
        # print('-------------------------------------------------- Dataset - ', i + 1, ' - 75%-Algorithm Comparison ',
        #       '--------------------------------------------------')
        # print(Table)
        #
        Table = PrettyTable()
        Table.add_column(Classifier[0], Terms)
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value[len(Algorithm) + j - 1, :])
        print('-------------------------------------------------- Dataset - ', i + 1, ' - 75%-Classifier Comparison',
              '--------------------------------------------------')
        print(Table)

    learnper = [35, 45, 55, 65, 75, 85]
    for i in range(1):
        for j in range(len(Graph_Term)):
            Graph = np.zeros(eval.shape[1:3])
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    if j == 9:
                        Graph[k, l] = eval[i, k, l, Graph_Term[j] + 4]
                    else:
                        Graph[k, l] = eval[i, k, l, Graph_Term[j] + 4]

            plt.plot(learnper, Graph[:, 0], color='r', linestyle='dashed', linewidth=3, marker='v', markerfacecolor='b', markersize=16,
                     label="EHO-OEL")
            plt.plot(learnper, Graph[:, 1], color='g', linestyle='dashed', linewidth=3, marker='s', markerfacecolor='red', markersize=12,
                     label="DOX-OEL")
            plt.plot(learnper, Graph[:, 2], color='b', linestyle='dashed', linewidth=3, marker='>', markerfacecolor='green', markersize=16,
                     label="HLBO-OEL")
            plt.plot(learnper, Graph[:, 3], color='c', linestyle='dashed', linewidth=3, marker='D', markerfacecolor='cyan', markersize=12,
                     label="PFA-OEL")
            plt.plot(learnper, Graph[:, 4], color='k', linestyle='dashed', linewidth=3, marker='p', markerfacecolor='black', markersize=16,
                     label="SPHLP-OEL")
            plt.xlabel('Training Set Percentage (%)')
            plt.ylabel(Terms[Graph_Term[j]])
            # plt.ylim([80, 100])
            plt.legend(loc=4)
            path1 = "./Results/Dataset_%s_%s_line.png" % (i + 1, Terms[Graph_Term[j]])
            plt.savefig(path1)
            plt.show()

            fig = plt.figure()
            # ax = plt.axes(projection="3d")
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            X = np.arange(6)
            ax.bar(X + 0.00, Graph[:, 5], color='r', width=0.10, label="SVM")
            ax.bar(X + 0.10, Graph[:, 6], color='g', width=0.10, label="ANN")
            ax.bar(X + 0.20, Graph[:, 7], color='b', width=0.10, label="ADABOOST")
            ax.bar(X + 0.30, Graph[:, 8], color='m', width=0.10, label="LOGISTIC REGRESSION")
            ax.bar(X + 0.40, Graph[:, 9], color='y', width=0.10, label="ENSEMBLE")
            ax.bar(X + 0.50, Graph[:, 10], color='k', width=0.10, label="SPHLP-OEL")
            plt.xticks(X + 0.10, ('35', '45', '55', '65', '75', '85'))
            plt.xlabel('Training Set Percentage (%)')
            plt.ylabel(Terms[Graph_Term[j]])
            plt.legend(loc=1)
            path1 = "./Results/Dataset_%s_%s_bar.png" % (i + 1, Terms[Graph_Term[j]])
            plt.savefig(path1)
            plt.show()


def stats(val):
    v = np.zeros(5)
    v[0] = max(val)
    v[1] = min(val)
    v[2] = np.mean(val)
    v[3] = np.median(val)
    v[4] = np.std(val)
    return v


def plot_Convergence():
    # matplotlib.use('TkAgg')
    Result = np.load('Fitness.npy', allow_pickle=True)
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'EHO', 'DOX', 'HLBO', 'PFA', 'PROPOSED']
    # Data = ['Psoriasis', 'Vitiligo']

    for i in range(Result.shape[0]):
        Terms = ['Worst', 'Best', 'Mean', 'Median', 'Std']
        Conv_Graph = np.zeros((5, 5))
        for j in range(5):  # for 5 algms
            Conv_Graph[j, :] = stats(Fitness[i, j, :])
            # a = 1
        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('-------------------------------------------------- ', 'Statistical Report ',
              '--------------------------------------------------')

        print(Table)

        length = np.arange(25)
        Conv_Graph = Fitness[i]
        # Conv_Graph = np.reshape(BestFit[i], (8, 20))
        plt.plot(length, Conv_Graph[0, :], color='k', linewidth=3, marker='o', markerfacecolor='red', markersize=12,
                 label='EHO-OEL')
        plt.plot(length, Conv_Graph[1, :], color='k', linewidth=3, marker='o', markerfacecolor='green',
                 markersize=12,
                 label='DOX-OEL')
        plt.plot(length, Conv_Graph[2, :], color='k', linewidth=3, marker='o', markerfacecolor='blue',
                 markersize=12,
                 label='HLBO-OEL')
        plt.plot(length, Conv_Graph[3, :], color='k', linewidth=3, marker='o', markerfacecolor='magenta',
                 markersize=12,
                 label='PFA-OEL')
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='o', markerfacecolor='k',
                 markersize=12,
                 label='SPHLP-OEL')
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        plt.savefig("./Results/Dataset_%s_%s_Conv.png" % (i + 1, 1))
        plt.show()


if __name__ == '__main__':
    # plot_results_1()
    # plot_Convergence()
    plot_results()
