#! /usr/bin/env
# -*- coding:utf-8 -*-

import csv
import numpy as np
from sklearn.metrics import r2_score

'''
  Metrics, 
  masterqkk@outlook.com
'''


def mae(true, pred):
    return np.mean(np.abs(true - pred))


def mape(true, pred):
    return np.mean(np.abs(true - pred) / true)


def mse(true, pred):
    return np.mean(np.power(true-pred, 2))


def me(true, pred):
    return np.mean(true - pred)


def std(true, pred):
    return np.sqrt(np.mean(np.power(true-pred, 2)))


def r2(true, pred):
    return r2_score(true, pred)


def write_csv_file(file_name, field_names, data_dict_list):
    '''
    :param file_name:
    :param field_names:
    :param data_dict_list: [dict, dict, dict], each dict corresponds to a row in table
    :return:
    '''
    with open(file_name, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)

        writer.writeheader()
        for row in data_dict_list:
            writer.writerow(row)

    print('write file {} finished.'.format(file_name))