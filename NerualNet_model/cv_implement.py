import numpy as np
import nn_implement


def generate_cv_data(X, y, k):
    m = X.shape[0]
    idx = 0
    X_cv = []
    y_cv = []
    for i in range(k):
        Xtmp = X[idx: idx + m // k - 1, :]
        ytmp = y[idx: idx + m // k - 1]
        X_cv.append(Xtmp)
        y_cv.append(ytmp)
        idx = idx + m // k - 1

    return X_cv, y_cv


def form_cv_data(X, y, k, cross_fold):
    # cyclic_list = np.ones((1, cross_fold))
    cyclic_list = [int(1)] * cross_fold
    cyclic_list[k] = 0
    xc_train = []
    yc_train = []
    xc_test = []
    yc_test = []
    for i in range(cross_fold):
        if cyclic_list[i] == 1:
            xc_train.append(X[i])
            yc_train.append(y[i])

        else:
            xc_test.append(X[i])
            yc_test.append(y[i])

    # formulate big matrix
    xct = xc_train[0]
    yct = yc_train[0]
    for i in range(1, len(xc_train)):
        xct = np.vstack((xct, xc_train[i]))
        yct = np.vstack((yct, yc_train[i]))

    # return value
    return xct, yct, xc_test[0], yc_test[0]


def do_cross_validation(xcv, ycv, cross_fold, neu, act_fn, batch_size):
    weights_cv = []
    score_cv = []
    binacc_cv = []
    configs_cv = []
    for i in range(cross_fold):
        print("\tvalidation ", i+1, " out of ", cross_fold, "...")
        [xc_train, yc_train, xc_test, yc_test] = form_cv_data(xcv, ycv, i, cross_fold)
        data_list = [xc_train, yc_train, xc_test, yc_test]
        [_, score, binacc, _, _, weights, configs] = nn_implement.do_neuron_network(neu, act_fn, batch_size, data_list)
        weights_cv.append(weights)
        score_cv.append(score)
        binacc_cv.append(binacc)
        configs_cv.append(configs)

    # return the highest prediction accuracy
    max_idx = np.argmin(np.asarray(score_cv))

    return weights_cv[max_idx], score_cv[max_idx], binacc_cv[max_idx], configs_cv[max_idx]
