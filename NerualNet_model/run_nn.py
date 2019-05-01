"""
This script implement the pipeline of converting original data into NN
and then predict the trend
Notice that this script manipulate data in the PCA data
"""

import numpy as np
import csv
import random
import nn_implement
import cv_implement


def do_pca_nn(filename):
    # implement PCA to refine features
    raw_data = []
    attr = []
    filepath = 'project_data_new/pca/' + filename + '_pca.csv'
    # filename = 'project_data_new/pca/amd_pca.csv'
    with open(filepath) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        iter = 0
        for row in readCSV:
            if iter == 0:
                attr.append(row)
            else:
                raw_data.append(row)
            iter = iter + 1

    m = len(raw_data)
    d = len(attr[0])
    X = np.zeros((m, d))    # data
    y = np.zeros((m, 1))    # label

    for i in range(m):
        for j in range(d):
            if raw_data[i][j] is '':
                X[i][j] = 0.0
            else:
                X[i][j] = float(raw_data[i][j])
        if raw_data[i][-1] is '':
            y[i] = int(0)
        else:
            y[i] = int(float(raw_data[i][-1]))

    X = X[:, 0:-1]
    fea_num = d - 1   # real number of features, the raw data contains the trend in the last column

    # split training and test data; choose random 10% as test data; generate cross validation data
    size_test = int(m*0.1)
    data_idx = np.arange(m-1)
    test_idx = random.sample(range(0, m-1), size_test)
    train_idx = np.delete(data_idx, test_idx)

    X_test = X[test_idx, :]
    y_test = y[test_idx, :]
    X_train = X[train_idx, :]
    y_train = y[train_idx, :]

    cross_fold = 5
    [X_cv, y_cv] = cv_implement.generate_cv_data(X_train, y_train, cross_fold)


    # run NN cross-validation
    cv_layer = [1, 2, 3, 5, 8, 10, 12, 15]
    cv_neuron = [100, 150, 200, 250, 300, 350]
    batch_size = 128

    # do cross validation
    cv_weight = []
    cv_score = []
    cv_binacc = []
    cv_config = []
    test_score = []
    test_binacc = []
    train_score = []
    train_binacc = []

    for i in range(len(cv_layer)):
        tmp_cv_weight = []
        tmp_cv_score = []
        tmp_cv_binacc = []
        tmp__cv_config = []

        tmp_train_score = []
        tmp_train_acc = []
        tmp_test_score = []
        tmp_test_acc = []
        for j in range(len(cv_neuron)):
            neu = [ cv_neuron[j] ] * cv_layer[i] + [1]
            act_fn = ['relu'] * cv_layer[i] + ['sigmoid']

            [w, s, bc, conf] = cv_implement.do_cross_validation(X_cv, y_cv, cross_fold, neu, act_fn, batch_size)
            tmp_cv_weight.append(w)
            tmp_cv_score.append(s)
            tmp_cv_binacc.append(bc)
            tmp__cv_config.append(conf)

            data_list = [X_train, y_train, X_test, y_test]
            [_, score, acc, score_train, acc_train, weights, configs] = nn_implement.do_neuron_network(neu, act_fn, batch_size, data_list)
            tmp_train_score.append(score_train)
            tmp_train_acc.append(acc_train)
            tmp_test_score.append(score)
            tmp_test_acc.append(acc)

        cv_weight.append(tmp_cv_weight)
        cv_score.append(tmp_cv_score)
        cv_binacc.append(tmp_cv_binacc)
        cv_config.append(tmp__cv_config)
        test_score.append(tmp_test_score)
        test_binacc.append(tmp_test_acc)
        train_score.append(tmp_train_score)
        train_binacc.append(tmp_train_acc)


    # choose highest prediction accuracy
    min_idx = np.unravel_index( np.argmin(np.asarray(cv_score)), np.shape(np.asarray(cv_score)) )
    print("Optimal parameters: layer = ", cv_layer[min_idx[0]], " neuron = ", cv_neuron[min_idx[1]], "\n")
    print("test_score = ", test_score[min_idx[0]][min_idx[1]], "\n")
    print("test_bin_acc = ", test_binacc[min_idx[0]][min_idx[1]], "\n")

    # store training, test and cross validation data
    filepath = 'project_data_new/results/pca/' + filename + '/'
    np.savetxt(filepath+"cv.csv", np.asarray(cv_score), delimiter=",")
    np.savetxt(filepath+"train.csv", np.asarray(train_score), delimiter=",")
    np.savetxt(filepath+"test.csv", np.asarray(test_score), delimiter=",")

