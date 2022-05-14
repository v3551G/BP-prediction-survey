#! /usr/bin/env
# -*- coding:utf-8 -*-

import os
import csv
import pandas as pd
import numpy as np

from analyze.tools import mae, me, std, mse, mape, r2, write_csv_file


'''
  Results analyzing tools,
  masterqkk@outlook.com
'''

base_dir = '../results'
# split_strategy = 'sir'  # 'r'/'s'/'si'/'sir'

split_strategy = 'r_3.0' # 'r_0.5'/'r'/ 'r_2.0'/ 'r_3.0'/ 'r_4.0'

phase_list = ['train', 'val', 'test']
random_seed_list = [0, 1, 2, 3, 4]  # each experiment is repeated 5 times.

for phase in phase_list:
    base_file_path = os.path.join(base_dir, split_strategy)

    SBP_results_list, DBP_results_list = [], []
    for random_seed in random_seed_list:
        file_path = os.path.join(base_file_path, str(random_seed))

        files = os.listdir(file_path)
        files = [x for x in files if x.endswith('.csv') and '{}_results'.format(phase) in x]

        df = pd.read_csv(os.path.join(file_path, files[0]))
        assert ('SBP_true' in df.columns and 'SBP_est' in df.columns and 'DBP_true' in df.columns and 'DBP_est' in df.columns)

        data = df.values

        SBP_true = df['SBP_true'].values
        SBP_est = df['SBP_est'].values
        DBP_true = df['DBP_true'].values
        DBP_est = df['DBP_est'].values

        mae_sbp, me_sbp, std_sbp, mse_sbp, mape_sbp, r2_sbp = mae(SBP_true, SBP_est), me(SBP_true, SBP_est), std(SBP_true, SBP_est), \
                                                              mse(SBP_true, SBP_est), mape(SBP_true, SBP_est), r2(SBP_true, SBP_est)
        mae_dbp, me_dbp, std_dbp, mse_dbp, mape_dbp, r2_dbp = mae(DBP_true, DBP_est), me(DBP_true, DBP_est), std(DBP_true, DBP_est), \
                                                              mse(DBP_true, DBP_est), mape(DBP_true, DBP_est), r2(DBP_true, DBP_est)

        data_dict_list = [{'task': 'sbp', 'mae': mae_sbp, 'mape': mape_sbp, 'mse': mse_sbp, 'me': me_sbp, 'std': std_sbp, 'r2': r2_sbp},
                          {'task': 'dbp', 'mae': mae_dbp, 'mape': mape_dbp, 'mse': mse_dbp, 'me': me_dbp, 'std': std_dbp, 'r2': r2_dbp}]
        field_names = ['task', 'mae', 'mape', 'mse', 'me', 'std', 'r2']

        print('Experimnet {}:\n SBP metrics, mae: {:.3f}, mape: {:.3f}, mse: {:.3f}, me: {:.3f}, std: {:.3f}, r2: {:.3f}'.format(
            random_seed, mae_sbp, mape_sbp, mse_sbp, me_sbp, std_sbp, r2_sbp))
        print('DBP metrics, mae: {:.3f}, mape: {:.3f}, mse: {:.3f}, me: {:.3f}, std: {:.3f}, r2: {:.3f}'.format(
            mae_dbp, mape_dbp, mse_dbp, me_dbp, std_dbp, r2_dbp))

        SBP_results_list.append([mae_sbp, mape_sbp, mse_sbp, me_sbp, std_sbp, r2_sbp])
        DBP_results_list.append([mae_dbp, mape_dbp, mse_dbp, me_dbp, std_dbp, r2_dbp])

    # summary statistics
    SBP_results = np.array(SBP_results_list)
    DBP_results = np.array(DBP_results_list)

    SBP_results_mean, SBP_results_std = np.mean(SBP_results, axis=0), np.std(SBP_results, axis=0)
    DBP_results_mean, DBP_results_std = np.mean(DBP_results, axis=0), np.std(DBP_results, axis=0)

    print('SBP results, \n mean: {}, \n std: {}'.format(SBP_results_mean, SBP_results_std))
    print('DBP results, \n mean: {}, \n std: {}'.format(DBP_results_mean, DBP_results_std))

    file_name = '{}-{}'.format(files[0].split('/')[-1].split('.')[0], split_strategy)
    #write_csv_file(os.path.join('.', file_name), field_names, data_dict_list)




