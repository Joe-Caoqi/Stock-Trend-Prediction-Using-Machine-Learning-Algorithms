"""
This script implements NN to preform binary classification
"""

from keras.models import Sequential
from keras.layers import Dense
import time
import random


def generate_batch_data(data_sp, label, batch_size):
    m = data_sp.shape[0]
    while 1:
        randIndex = random.sample(range(0, m), batch_size)  # create a random index set to choose data randomly
        #data = data_sp[randIndex, :].toarray()
        data = data_sp[randIndex, :]
        yield (data, label[randIndex,:])
        data = []


def do_neuron_network(neu, act_fn, batch_size, data_list):
    # read data
    X_train = data_list[0]
    y_train = data_list[1]
    X_test = data_list[2]
    y_test = data_list[3]
    layer = len(neu)
    m = X_train.shape[0]
    d = X_train.shape[1]


    # define model
    model = Sequential()
    model.add(Dense(units=neu[0], activation=act_fn[0], input_dim=d))
    for i in range(1, layer, 1):
        model.add(Dense(units=neu[i], activation=act_fn[i]))

    model.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=['binary_accuracy'])


    t_start = time.time()
    model.fit_generator(generate_batch_data(X_train, y_train, batch_size),
                        steps_per_epoch=m // batch_size,
                        epochs=8,
                        verbose=0,
                        validation_data=(X_test, y_test))

    t_elapse = time.time() - t_start
#    print("training time = ", t_elapse)

    score_train = model.evaluate(x=X_train, y=y_train, verbose=0)
#    print("training result: ", score_train)
    score = model.evaluate(x=X_test, y=y_test, verbose=0)
    print("test result: ", score)

    return t_elapse, score[0], score[1], score_train[0], score_train[1], model.get_weights(), model.get_config()

