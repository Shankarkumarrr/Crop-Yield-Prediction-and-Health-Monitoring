import cv2 as cv
import numpy as np
# import tflearn
# from tensorflow.python.framework import ops
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from torch import ops
from Evaluation import evaluation


def Model_3DCNN(Data, Target):
    IMG_SIZE = 20
    Train_X = np.zeros((Data.shape[0], IMG_SIZE, IMG_SIZE, 1))
    for i in range(Data.shape[0]):
        temp = np.resize(Data[i], (IMG_SIZE * IMG_SIZE, 1))
        Train_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 1))

    Test_X = np.zeros((Data.shape[0], IMG_SIZE, IMG_SIZE, 1))
    for i in range(Data.shape[0]):
        temp = np.resize(Data[i], (IMG_SIZE * IMG_SIZE, 1))
        Test_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 1))
    pred, weight = Model(Data, Target)

    pred = np.asarray(pred)
    Eval = evaluation(pred, Target)
    feat = np.asarray(weight)
    feat = np.reshape(feat, (feat.shape[0] * feat.shape[1], feat.shape[2] * feat.shape[3]))
    feat = np.resize(feat, (Data.shape[0] + Data.shape[0], 1000))

    return Eval, feat


def Model_CNN(train_data, train_target, test_data, test_target, sol=None):
    if sol is None:
        sol = [128, 0.01, 5]
    IMG_SIZE = 20
    Train_X = np.zeros((train_data.shape[0], IMG_SIZE, IMG_SIZE, 1))
    for i in range(train_data.shape[0]):
        temp = np.resize(train_data[i], (IMG_SIZE * IMG_SIZE, 1))
        Train_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 1))

    Test_X = np.zeros((test_data.shape[0], IMG_SIZE, IMG_SIZE, 1))
    for i in range(test_data.shape[0]):
        temp = np.resize(test_data[i], (IMG_SIZE * IMG_SIZE, 1))
        Test_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 1))
    pred = Model(Train_X, train_target, Test_X, test_target, sol)

    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    Eval = evaluation(pred, test_target)
    return Eval, pred


def Model(X, Y, test_x, test_y, sol):
    LR = 1e-3
    ops.reset_default_graph()
    convnet = input_data(shape=[None, 20, 20, 1], name='input')

    convnet = conv_2d(convnet, 32, 5, name='layer-conv1', activation='linear')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, name='layer-conv2', activation='linear')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, round(sol[0]), 5, name='layer-conv3', activation='linear')
    convnet = max_pool_2d(convnet, 5)

    convnetc = conv_2d(convnet, 64, 5, name='layer-conv4', activation='linear')
    convnet = max_pool_2d(convnetc, 5)

    convnet = conv_2d(convnet, 32, 5, name='layer-conv5', activation='linear')
    convnet = max_pool_2d(convnet, 5)

    convnet1 = fully_connected(convnet, 1024, name='layer-conv', activation='linear')
    convnet2 = dropout(convnet1, 0.8)

    convnet3 = fully_connected(convnet2, Y.shape[1], name='layer-conv-before-softmax', activation='linear')

    regress = regression(convnet3, optimizer='sgd', learning_rate=sol[2],
                         loss='mean_square', name='target')

    model = tflearn.DNN(regress, tensorboard_dir='log')

    MODEL_NAME = 'test.model'.format(LR, '6conv-basic')
    model.fit({'input': X}, {'target': Y}, n_epoch=round(sol[1]),
              validation_set=({'input': test_x}, {'target': test_y}),
              snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

    pred = model.predict(test_x)
    weight = model.get_weights(convnetc.W)
    return pred, weight

def Model_3DCNN(Data, Target):
    # IMG_SIZE = 20
    # Train_X = np.zeros((Data.shape[0], IMG_SIZE, IMG_SIZE, 1))
    # for i in range(Data.shape[0]):
    #     temp = np.resize(Data[i], (IMG_SIZE * IMG_SIZE, 1))
    #     Train_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 1))
    #
    # Test_X = np.zeros((Data.shape[0], IMG_SIZE, IMG_SIZE, 1))
    # for i in range(Data.shape[0]):
    #     temp = np.resize(Data[i], (IMG_SIZE * IMG_SIZE, 1))
    #     Test_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 1))
    weight = Model_3d(Data, Target)

    # pred = np.asarray(pred)
    # Eval = evaluation(pred, Target)
    feat = np.asarray(weight)
    feat = np.reshape(feat, (feat.shape[0] * feat.shape[1], feat.shape[2] * feat.shape[3]))
    feat = np.resize(feat, (Data.shape[0] + Data.shape[0], 1000))

    return feat

def Model_3d(data, target):
    LR = 1e-3
    # ops.reset_default_graph()
    convnet = input_data(shape=[64, 512, 512], name='input')

    convnet = conv_2d(convnet, 32, 5, name='layer-conv1', activation='linear')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, name='layer-conv2', activation='linear')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 5, 5, name='layer-conv3', activation='linear')
    convnet = max_pool_2d(convnet, 5)

    convnetc = conv_2d(convnet, 64, 5, name='layer-conv4', activation='linear')
    convnet = max_pool_2d(convnetc, 5)

    convnet = conv_2d(convnet, 32, 5, name='layer-conv5', activation='linear')
    convnet = max_pool_2d(convnet, 5)

    convnet1 = fully_connected(convnet, 1024, name='layer-conv', activation='linear')
    convnet2 = dropout(convnet1, 0.8)

    convnet3 = fully_connected(convnet2, data.shape[1], name='layer-conv-before-softmax', activation='linear')

    regress = regression(convnet3, optimizer='sgd', learning_rate=0.03,
                         loss='mean_square', name='target')

    model = tflearn.DNN(regress, tensorboard_dir='log')

    MODEL_NAME = 'test.model'.format(LR, '6conv-basic')
    model.fit({'input': data}, {'target': target}, n_epoch=1,
              validation_set=({'input': data}, {'target': target}),
              snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

    # pred = model.predict(data)
    weight = model.get_weights(convnetc.W)
    return weight