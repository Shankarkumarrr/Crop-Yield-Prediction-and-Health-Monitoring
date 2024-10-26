import numpy as np
import tensorflow as tf
from keras import activations
from tensorflow.keras.layers import Flatten, Dense, Conv1D, MaxPool1D, Dropout
# import tensorflow.compat.v1 as tf
from Evaluation import evaluation


def Model_3DCNN(Data, Target, sol=None):
    if sol is None:
        sol = [128, 5]

    IMG_SIZE = 20
    Train_X = np.reshape(Data, (Data.shape[0], Data.shape[1], 512))
    # Target = np.reshape(Target, (Target.shape[0], 1))

    # Test_X = np.zeros((test_data.shape[0], IMG_SIZE, IMG_SIZE, 1))
    # for i in range(test_data.shape[0]):
    #     temp = np.resize(test_data[i], (IMG_SIZE * IMG_SIZE, 1))
    #     Test_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 1))
    activation = Model(Train_X, Target, sol)

    # pred = np.asarray(pred)
    feat = np.asarray(activation)

    return activation


def Model(X, Y, sol):
    batch = 16
    epochs = 10
    shape = np.size(X, 1)

    cnn_model = tf.keras.models.Sequential()
    # First CNN layer  with 32 filters, conv window 3, relu activation and same padding
    cnn_model.add(
        Conv1D(filters=32, kernel_size=(3,), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.001),
               input_shape=(X.shape[1], 1)))
    # Second CNN layer  with 64 filters, conv window 3, relu activation and same padding
    cnn_model.add(
        Conv1D(filters=64, kernel_size=(3,), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.001)))
    # Third CNN layer with 128 filters, conv window 3, relu activation and same padding
    cnn_model.add(
        Conv1D(filters=int(sol[0]), kernel_size=(3,), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.001)))
    # Fourth CNN layer with Max pooling
    cnn_model.add(MaxPool1D(pool_size=(3,), strides=2, padding='same'))
    cnn_model.add(Dropout(0.5))
    # Flatten the output
    cnn_model.add(Flatten())
    # Add a dense layer with 256 neurons
    cnn_model.add(Dense(units=256, activation=tf.keras.layers.LeakyReLU(alpha=0.001)))
    # Add a dense layer with 512 neurons
    cnn_model.add(Dense(units=512, activation=tf.keras.layers.LeakyReLU(alpha=0.001)))
    # Softmax as last layer with five outputs
    cnn_model.add(Dense(units=5, activation='softmax'))
    cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    cnn_model.summary()

    # cnn_model_history = cnn_model.fit(X, Y, epochs=1, batch_size=10, validation_data=(X, Y))

    # pred = cnn_model.predict(X)
    activation = activations.relu(X)  # .numpy()
    # activation = activation.eval(session=tf.compat.v1.Session())
    activation = activation.numpy()
    feat = np.asarray(activation)
    feat = np.reshape(feat, (feat.shape[0], feat.shape[1] * feat.shape[2]))
    feat = np.resize(feat, (X.shape[0], 1000))
    return feat
# array = tf.Session().run(tensor)
