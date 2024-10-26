import numpy as np
from sklearn.ensemble import AdaBoostClassifier

from Evaluation import evaluation


def Model_Adaboost(train_data, train_target, test_data, test_target, sol):
    sol = int(sol)
    model = AdaBoostClassifier(n_estimators=sol, learning_rate = sol, random_state=0)
    pred = np.zeros(test_target.shape)
    for i in range(test_target.shape[1]):
        model.fit(train_data, train_target[:, i])
        pred[:, i] = model.predict(test_data)

    Eval = evaluation(pred, test_target)
    return Eval