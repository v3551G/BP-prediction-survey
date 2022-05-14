#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
from os.path import expanduser, join, exists, isdir
from os import environ, makedirs
from sys import argv
from functools import partial
from datetime import datetime
from copy import deepcopy
from matplotlib import pyplot as plt
# from multiprocessing import Pool

import argparse
import seaborn as sns
import tensorflow as tf
import pandas as pd
import numpy as np

from models.define_ResNet_1D import ResNet50_1D


'''
    Train a BP predictive model using PPG signal. 
    checkpoint callbacks are used to store the network weights and the best network weights in terms of validation loss during training, respectively. 
    The best model is then used for evaluation on the test set, and the original test results are stored in a csv file for further analysis.
    
    Note that the code supports continuing training after interruption.
    
    masterqkk@outlook.com
'''


def read_tfrecord(example, win_len=875):
    tfrecord_format = (
        {
            'ppg': tf.io.FixedLenFeature([win_len], tf.float32),
            'label': tf.io.FixedLenFeature([2], tf.float32)
        }
    )
    parsed_features = tf.io.parse_single_example(example, tfrecord_format)

    return parsed_features['ppg'], (parsed_features['label'][0], parsed_features['label'][1])


def create_dataset(tfrecords_dir, tfrecord_basename, win_len=875, batch_size=32, modus='train'):
    '''
    Collect the corresponding data files and wrap the data into a standard TFRecordDataset object for training.
    :param tfrecords_dir:
    :param tfrecord_basename:
    :param win_len:
    :param batch_size:
    :param modus: train/ val/ test
    :return:
    '''
    pattern = join(tfrecords_dir, modus, tfrecord_basename + "_" + modus + "_?????_of_?????.tfrecord")
    dataset = tf.data.TFRecordDataset.list_files(pattern)

    if modus == 'train':
        dataset = dataset.shuffle(1000, reshuffle_each_iteration=True)
        dataset = dataset.interleave(
            tf.data.TFRecordDataset,
            cycle_length=800,
            block_length=400)
    else:
        dataset = dataset.interleave(
            tf.data.TFRecordDataset)

    dataset = dataset.map(partial(read_tfrecord, win_len=win_len), num_parallel_calls=2)
    dataset = dataset.shuffle(4096, reshuffle_each_iteration=True)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.repeat()

    return dataset


def get_model(architecture, input_shape, UseDerivative=False):
    if architecture == 'resnet':
        return ResNet50_1D(input_shape, UseDerivative=UseDerivative)
    else:
        print('The model architecture {} has not been implemented.'.format(architecture))
        exit(-1)


def get_init_epoch(file_name):
    '''
    get initial epoch from the corresponding CSV file
    :param file_name:
    :return:
    '''
    with open(file_name) as f:
        f_csv = csv.DictReader(f)
        count = 0
        for row in f_csv:
            count += 1
        return count


def ppg_train_mimic_iii(architecture, DataDir, ResultsDir, CheckpointDir, tensorboard_tag, tfrecord_basename, experiment_name, split_strategy='r',
                        win_len=875, batch_size=32, lr = None, N_epochs = 20, Ntrain=1e6, Nval=2.5e5, Ntest=2.5e5, UseDerivative=False,
                        earlystopping=True, enlarge_ratio=1.0, random_seed=0, verbose=0):
    '''
    load the procssed data to train a ResNet model for BP estimation.
    :param architecture: model architecture, 'resnet'
    :param DataDir:
    :param ResultsDir:
    :param CheckpointDir:
    :param tensorboard_tag:
    :param tfrecord_basename:
    :param experiment_name:
    :param split_strategy: splitting strategy, 'r'/ 'ru'/ 'sa'/ 'si' / 'sir', for the detail meaning please refer '3split_and_save_dataset.py'.
    :param win_len:
    :param batch_size:
    :param lr: learning rate.
    :param N_epochs: the maximum number of trainnig epochs.
    :param Ntrain: the number of training samples.
    :param Nval: the number of validating samples.
    :param Ntest: the number of test samples.
    :param UseDerivative: whether use PPG derivatives.
    :param earlystopping: whether use early stopping strategy
    :param enlarge_ratio: determine the size of the collected dataset for training, set 1.0 in default, which corresponds to the 'v1' version mentioned in the manuscript.
    :param random_seed:
    :param verbose:
    :return:
    '''
    # Extend dir based on split_strategy
    assert (split_strategy is not None)

    if enlarge_ratio == 1.0:
        split_strategy_ext = split_strategy
    else:
        split_strategy_ext = split_strategy + '_' + str(enlarge_ratio)

    DataDir = join(DataDir, split_strategy_ext, str(random_seed))

    print('DataDir: {}'.format(DataDir))

    ResultsDir = join(ResultsDir, split_strategy_ext, str(random_seed))
    CheckpointDir = join(CheckpointDir, split_strategy_ext, str(random_seed))

    if not exists(DataDir) or not isdir(DataDir):
        makedirs(DataDir, exist_ok=True)

    if not exists(ResultsDir) or not isdir(ResultsDir):
        makedirs(ResultsDir, exist_ok=True)

    if not exists(CheckpointDir) or not isdir(CheckpointDir):
        makedirs(CheckpointDir, exist_ok=True)

    # create datasets for training, validation and testing using .tfrecord files
    test_dataset = create_dataset(DataDir, tfrecord_basename, win_len=win_len, batch_size=batch_size,
                                  modus='test')
    train_dataset = create_dataset(DataDir, tfrecord_basename, win_len=win_len, batch_size=batch_size, modus='train')
    val_dataset = create_dataset(DataDir, tfrecord_basename, win_len=win_len, batch_size=batch_size,
                                 modus='val')

    data_in_shape = (win_len, 1)

    # load the neuro architecture
    model: object = get_model(architecture, data_in_shape, UseDerivative=UseDerivative)

    # callback for logging training and validation results
    csvLogger_cb = tf.keras.callbacks.CSVLogger(
        filename=join(ResultsDir, experiment_name + '_learningcurve.csv')
    )

    # checkpoint callback
    checkpoint_cb_best = tf.keras.callbacks.ModelCheckpoint(
        filepath=join(CheckpointDir, 'best', experiment_name + '_cb.h5'),
        save_best_only=True,
        save_weights_only=False
    )

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=join(CheckpointDir, experiment_name + '_cb.h5'),
        save_weights_only=False
    )

    # tensorboard callback
    tensorbard_cb = tf.keras.callbacks.TensorBoard(
        log_dir=join(ResultsDir, 'tb', tensorboard_tag),
        histogram_freq=0,
        write_graph=False
    )

    # callback for early stopping if validation loss stops improving
    EarlyStopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    # define Adam optimizer
    if lr is None:
        opt = tf.keras.optimizers.Adam()
    else:
        opt = tf.keras.optimizers.Adam(learning_rate=lr)

    # compile model using mean squared error as loss function
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.mean_squared_error,
        metrics=[['mae'], ['mae']]
    )

    # determine the start point of training
    init_epoch = 0
    hist_csv_log_files = [x for x in os.listdir(ResultsDir) if x.endswith('csv') and 'learningcurve' in x]
    if len(hist_csv_log_files) > 0:
        init_epoch = get_init_epoch(join(ResultsDir, hist_csv_log_files[0]))

    if init_epoch < N_epochs:
        # load the latest checkpoint for continuously training
        model_exists = os.listdir(CheckpointDir)
        model_exists = [file for file in model_exists if file.endswith('.h5') and architecture in file]
        if len(model_exists) > 0:
            model.load_weights(join(CheckpointDir, model_exists[-1]))

        cb_list = [checkpoint_cb,
                   checkpoint_cb_best,
                   tensorbard_cb,
                   csvLogger_cb,
                   EarlyStopping_cb if earlystopping == True else []]

        # Perform Training and Validation, modify 20220322.
        history = model.fit(
            train_dataset,
            steps_per_epoch=int(Ntrain//batch_size),
            epochs=N_epochs,
            validation_data=val_dataset,
            validation_steps=int(Nval//batch_size),
            initial_epoch=init_epoch,
            callbacks=cb_list,
            workers=2,
            use_multiprocessing=True,
            verbose=verbose
        )
    else:
        # load the best model for prediction
        model.load_weights(checkpoint_cb_best.filepath)

    # Predictions on the testset
    test_results = pd.DataFrame({
        'SBP_true' : [],
        'DBP_true' : [],
        'SBP_est' : [],
        'DBP_est' : []
    })
    train_results = deepcopy(test_results)
    val_results = deepcopy(test_results)

    # store predictions on the test set as well as the corresponding ground truth in a csv file
    test_dataset = iter(test_dataset)
    for i in range(int(Ntest//batch_size)):
        ppg_test, BP_true = test_dataset.next()
        #with tf.no_gradient():
        BP_est = model.predict(ppg_test)
        TestBatchResult = pd.DataFrame({
            'SBP_true' : BP_true[0].numpy(),
            'DBP_true' : BP_true[1].numpy(),
            'SBP_est' : np.squeeze(BP_est[0]),
            'DBP_est' : np.squeeze(BP_est[1])
        })
        test_results = test_results.append(TestBatchResult)

    TestResultsFile = join(ResultsDir, experiment_name + '_test_results.csv')
    test_results.to_csv(TestResultsFile)

    # performed for visualizing error distributions among training set, validation set and test set, 20220222.
    train_dataset = iter(train_dataset)
    for i in range(int(Ntrain // batch_size)):
        ppg_train, BP_true = train_dataset.next()
        #with tf.no_gradient():
        BP_est = model.predict(ppg_train)
        TrainBatchResult = pd.DataFrame({
            'SBP_true': BP_true[0].numpy(),
            'DBP_true': BP_true[1].numpy(),
            'SBP_est': np.squeeze(BP_est[0]),
            'DBP_est': np.squeeze(BP_est[1])
        })
        train_results = train_results.append(TrainBatchResult)

    TrainResultsFile = join(ResultsDir, experiment_name + '_train_results.csv')
    train_results.to_csv(TrainResultsFile)

    val_dataset = iter(val_dataset)
    for i in range(int(Nval // batch_size)):
        ppg_val, BP_true = val_dataset.next()
        #with tf.no_gradient():
        BP_est = model.predict(ppg_val)
        ValBatchResult = pd.DataFrame({
            'SBP_true': BP_true[0].numpy(),
            'DBP_true': BP_true[1].numpy(),
            'SBP_est': np.squeeze(BP_est[0]),
            'DBP_est': np.squeeze(BP_est[1])
        })
        val_results = val_results.append(ValBatchResult)

    ValResultsFile = join(ResultsDir, experiment_name + '_val_results.csv')
    val_results.to_csv(ValResultsFile)

    # calculate metrics (mainly for bin error distribution plot)
    metrics_train, metrics_train_bin = cal_metrics(train_results.to_dict('list'))
    metrics_val, metrics_val_bin = cal_metrics(val_results.to_dict('list'))
    metrics_test, metrics_test_bin = cal_metrics(test_results.to_dict('list'))
    
    print('calculate metrics finished.')

    # Mean predictor, 20220326
    mae_sbp_mp, mae_dbp_mp = cal_metrics_mean_predictor(train_results.to_dict('list'), test_results.to_dict('list'))
    print('Mean predictor, MAE, SBP: {}, DBP: {}'.format(mae_sbp_mp, mae_dbp_mp))

    # plot error distribution,
    # reference: https://docs.python.org/zh-cn/3/library/multiprocessing.html
    #pool = Pool(3)
    #with pool:
    #    pool.map(plot_err_distr, [(metrics_train_bin, train_results, ResultsDir, architecture, 'train', 'MIMIC-III'),
    #                              (metrics_val_bin, val_results, ResultsDir, architecture, 'val', 'MIMIC-III'),
    #                              (metrics_test_bin, test_results, ResultsDir, architecture, 'test', 'MIMIC-III')]
    #            )

    plot_err_distr(metrics_train_bin, train_results, res_dir=ResultsDir, alg_name=architecture, phase='train')
    plot_err_distr(metrics_val_bin, val_results, res_dir=ResultsDir, alg_name=architecture, phase='val')
    plot_err_distr(metrics_test_bin, test_results, res_dir=ResultsDir, alg_name=architecture, phase='test')

    idx_min = np.argmin(history.history['val_loss'])

    print(' Training finished')

    return history.history['SBP_mae'][idx_min], history.history['DBP_mae'][idx_min], history.history['val_SBP_mae'][idx_min], history.history['val_DBP_mae'][idx_min]


def cal_metrics_mean_predictor(train_results, test_results):
    '''
    Calculate the prediction results of fixed mean predictor
    :param train_results:
    :param test_results:
    :return:
    '''
    SBP_train = train_results['SBP_true']
    DBP_train = train_results['DBP_true']
    SBP_test = test_results['SBP_true']
    DBP_test = test_results['DBP_true']

    num_test_smaples = len(SBP_test)

    SBP_pred = [np.mean(SBP_train)] * num_test_smaples
    DBP_pred = [np.mean(DBP_train)] * num_test_smaples

    mae_sbp = np.mean(np.abs([x-y for x, y in zip(SBP_pred, SBP_test)]))
    mae_dbp = np.mean(np.abs([x-y for x, y in zip(DBP_pred, DBP_test)]))

    return mae_sbp, mae_dbp


def cal_metrics(results):
    '''
    Calculate the test result of MAE on each BP interval
    :param results:
    :return:
    '''
    SBP_true = results['SBP_true']
    DBP_true = results['DBP_true']
    SBP_est  = results['SBP_est']
    DBP_est  = results['DBP_est']

    max_sbp, min_sbp = max(SBP_true), min(SBP_true)
    max_dbp, min_dbp = max(DBP_true), min(DBP_true)

    max_sbp_lim, min_sbp_lim = int(np.ceil(max_sbp / 10) * 10), int(np.floor(min_sbp / 10) * 10)
    max_dbp_lim, min_dbp_lim = int(np.ceil(max_dbp / 10) * 10), int(np.floor(min_dbp / 10) * 10)

    sbp_bin_err_dict, dbp_bin_err_dict = {}, {}
    # initialize bin error dict
    sbp_key_list = [str(min_sbp_lim + x * 10) + '-' + str(min_sbp_lim + (x+1) * 10) for x in range(int((max_sbp_lim - min_sbp_lim) // 10))]
    dbp_key_list = [str(min_dbp_lim + x * 10) + '-' + str(min_dbp_lim + (x+1) * 10) for x in range(int((max_dbp_lim - min_dbp_lim) // 10))]

    for key in sbp_key_list:
        sbp_bin_err_dict[key] = []

    for key in dbp_key_list:
        dbp_bin_err_dict[key] = []

    #print('sbp_bin_err_dict: {}'.format(sbp_bin_err_dict))
    #print('dbp_bin_err_dict: {}'.format(dbp_bin_err_dict))

    mae_sbp = np.abs([x - y for (x, y) in zip(SBP_est, SBP_true)])
    mae_dbp = np.abs([x - y for (x, y) in zip(DBP_est, DBP_true)])

    # print('SBP_true.shape: {}, \n{}'.format(SBP_true.shape, SBP_true))
    num_sam = len(SBP_true)

    for sid in range(num_sam):
        if SBP_true[sid] == max_sbp_lim:
            key_tmp = str(max_sbp_lim-10) + '-' + str(max_sbp_lim)
        else:
            key_tmp = str(int((SBP_true[sid] // 10) * 10)) + '-' + str(int((SBP_true[sid] // 10 + 1) * 10))

        sbp_bin_err_dict[key_tmp].append(mae_sbp[sid])

        if DBP_true[sid] == max_dbp_lim:
            key_tmp2 = str(max_dbp_lim-10) + '-' + str(max_dbp_lim)
        else:
            key_tmp2 = str(int((DBP_true[sid] // 10) * 10)) + '-' + str(int((DBP_true[sid] // 10 + 1) * 10))
        dbp_bin_err_dict[key_tmp2].append(mae_dbp[sid])

    return None,  (sbp_bin_err_dict, dbp_bin_err_dict)


def plot_err_distr(metrics_bin, results, res_dir, alg_name, phase, data_flag):
    '''
    Visualize the distribution of errors on different BP intervals.
    :param metrics_bin:
    :param results:
    :param res_dir:
    :param alg_name:
    :param phase: train/ val/ test
    :param data_flag:
    :return:
    '''

    (sbp_bin_err_dict, dbp_bin_err_dict) = metrics_bin
    sbp_true, dbp_true = results['SBP_true'], results['DBP_true']

    task_id_map = {0: 'SBP', 1: 'DBP', 2: 'MBP'}

    # wrap data
    table_sbp = pd.DataFrame(index=['alg'], columns=['alg', 'bin', 'error'])
    table_dbp = pd.DataFrame(index=['alg'], columns=['alg', 'bin', 'error'])

    sbp_key_list = list(sbp_bin_err_dict.keys())
    dbp_key_list = list(dbp_bin_err_dict.keys())

    for key in sbp_key_list:
        value_list = sbp_bin_err_dict[key]
        for elem in value_list:
            table_sbp.loc[len(table_sbp)+1] = {'alg': alg_name, 'bin': key, 'error': elem}

    for key in dbp_key_list:
        value_list = dbp_bin_err_dict[key]
        for elem in value_list:
            table_dbp.loc[len(table_dbp) + 1] = {'alg': alg_name, 'bin': key, 'error': elem}

    #
    table = [table_sbp, table_dbp]
    key_list = [sbp_key_list, dbp_key_list]
    gt = [sbp_true, dbp_true]
    num_task = 2

    for tid in range(num_task):
        label_list = key_list[tid]
        num_bins = len(label_list)

        fig, ((ax1), (ax2)) = plt.subplots(2, 1, figsize=(12, 8))
        sns.boxplot(ax=ax1, x="bin", y="error", hue="alg", showmeans=True, data=table[tid])
        ax1.grid(b=True, which='major', axis='y', color='k', linestyle='--', linewidth=0.5)
        # ax1.set_xlabel('')
        ax1.set_xticklabels(label_list, rotation=0, fontsize=8)
        ax1.set_ylabel('MAE (mmHg)')

        ax2.hist(gt[tid], bins=num_bins, range=(np.floor((min(gt[tid]) / 10)) * 10, np.ceil(max(gt[tid]) / 10) * 10))
        ax2.set_xlabel('BP interval (mmHg)')
        ax2.set_ylabel('Frequency')
        #
        fig_name = 'bin-error-distr_{}_{}_{}_{}'.format(alg_name, task_id_map[tid], data_flag, phase)
        plt.savefig(os.path.join(res_dir, fig_name) + '.svg', format='svg')
        plt.savefig(os.path.join(res_dir, fig_name) + '.jpg')


def set_gpu_memory_growth():
    # set memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        print('{}, memory_growth: {}'.format(gpu, tf.config.experimental.get_memory_growth(gpu)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ExpName', type=str, help="unique name for the training")
    parser.add_argument('datadir', type=str, help="folder containing the train, val and test subfolders containing tfrecord files")
    parser.add_argument('resultsdir', type=str, help="Directory in which results are stored")
    parser.add_argument('chkptdir', type=str, help="directory used for storing model checkpoints")

    parser.add_argument('--arch', type=str, default="resnet", help="neural architecture used for training (alexnet (default), resnet,  slapnicar, lstm)")
    parser.add_argument('--lr', type=float, default=0.001, help="initial learning rate (default: 0.003)")
    parser.add_argument('--batch_size', type=int, default=128, help="batch size used for training (default: 32)")
    parser.add_argument('--winlen', type=int, default=875, help="length of the ppg windows in samples (default: 875)")
    parser.add_argument('--epochs', type=int, default=50, help="maximum number of epochs for training (default: 60)")
    parser.add_argument('--gpuid', type=str, default='0, 1', help="GPU-ID used for training in a multi-GPU environment (default: None)")
    parser.add_argument('--enlarge_ratio', type=float, default=1.0, help='')
    parser.add_argument('--verbose', type=int, default=0, help='')

    parser.add_argument('--random_seed', type=int, default=0, help='')
    parser.add_argument('--split_strategy', type=str, default='r', help='')

    args = parser.parse_args()

    if len(argv) > 1:
        architecture = args.arch
        experiment_name = args.ExpName
        #experiment_name = datetime.now().strftime("%Y-%d-%m") + '_' + architecture + '_' + experiment_name
        experiment_name = architecture + '_' + experiment_name
        DataDir = args.datadir
        ResultsDir = args.resultsdir
        CheckpointDir = args.chkptdir
        tb_tag = experiment_name
        split_strategy = args.split_strategy
        batch_size = args.batch_size
        lr = (args.lr * batch_size) / 128 if args.lr > 0 else None  # linear scaling with batchsize, as in OLOX

        win_len = args.winlen
        N_epochs = args.epochs
        enlarge_ratio = args.enlarge_ratio
        random_seed = args.random_seed
        gpuid = args.gpuid
        verbose = args.verbose

        if split_strategy in ['s', 'si', 'sir', 'r']:
            Ntrain, Nval, Ntest = 900000, 300000, 300000
            if split_strategy == 'r' and enlarge_ratio in [0.5, 2.0, 4.0]:
                Ntrain, Nval, Ntest = int(900000 * enlarge_ratio), int(300000 * enlarge_ratio), int(300000 * enlarge_ratio)
        elif split_strategy == 'ru':
            print('The number of samples is too few')
            exit(-1)
        else:
            pass

        if gpuid is not None:
            environ["CUDA_VISIBLE_DEVICES"] = gpuid

        set_gpu_memory_growth()

    else:
        architecture = 'lstm'
        experiment_name = datetime.now().strftime("%Y-%d-%m") + '_' + architecture + '_' + 'mimic_iii_ppg_nonmixed_pretrain'
        HomePath = expanduser("~")
        DataDir = join(HomePath, 'data', 'MIMIC-III_BP', 'tfrecords_nonmixed')
        ResultsDir = join(HomePath, 'data', 'MIMIC-III_BP', 'results')
        CheckpointDir = join(HomePath, 'data', 'MIMIC-III_BP', 'checkpoints')
        tb_tag = architecture + '_' + 'mimic_iii_ppg_pretrain'
        batch_size = 64
        win_len = 875
        lr = None
        N_epochs = 60

    tfrecord_basename = 'MIMIC_III_ppg'

    ppg_train_mimic_iii(architecture, DataDir, ResultsDir, CheckpointDir, tb_tag, tfrecord_basename, experiment_name,
                        split_strategy=split_strategy, win_len=win_len, batch_size=batch_size, lr=lr, N_epochs=N_epochs,
                        Ntrain=Ntrain, Nval=Nval, Ntest=Ntest, UseDerivative=True, earlystopping=False, enlarge_ratio=enlarge_ratio,
                        random_seed=random_seed, verbose=verbose)