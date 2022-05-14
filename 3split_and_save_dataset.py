#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from datetime import datetime
from os import makedirs
from os.path import expanduser, isdir, join, exists
from sys import argv

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

"""
    Split the processed dataset into training, validation and test set and save as tfrecord format for further training.
    
    Specifically, this script reads a dataset consisting of PPG and BP samples from a .h5 file and converts them into a binary format that can be used for 
    as input data for a neural network during training. The dataset can be divided into training, validation and test set by different splitting strategies as follows.
    ——'r': record level， data of a record/subject appears only in training, validation, or test set.
    ——'ru': record level with single saple;
    ——'sa': sample level with aggregated operation;
    ——'si': sample level with intra record split;
    ——'sir': sample level with intra record random split;
    
    For the detail information about these splitting strategies, please refer the coming manuscript titled "Machine learning and deep learning for blood pressure prediction: A methodological review from
multiple perspectives".

    masterqkk@outlook.com, 20220320.
"""


def _float_feature(value):
    # Returns a float_list from a float / double
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def ppg_hdf2tfrecord(h5_file, tfrecord_path, samp_idx, weights_SBP=None, weights_DBP=None):
    '''
    converts PPG/BP sample pairs into the binary .tfrecord file format. This function creates a .tfrecord
    file containing a defined number os samples.
    :param h5_file: file containing ppg and BP data
    :param tfrecord_path: full path for storing the .tfrecord files
    :param samp_idx: sample indizes of the data in the .h5 file to be stored in the .tfrecord file
    :param weights_SBP: sample weights for the systolic BP (optional)
    :param weights_DBP: sample weights for the diastolic BP (optional)
    :return:
    '''

    N_samples = len(samp_idx)
    # open the .h5 file and get the samples with the indizes specified by samp_idx
    with h5py.File(h5_file, 'r') as f:
        # load ppg and BP data as well as the subject numbers the samples belong to
        ppg_h5 = f.get('/ppg')
        BP = f.get('/label')
        subject_idx = f.get('/subject_idx')

        writer = tf.io.TFRecordWriter(tfrecord_path)

        # iterate over each sample index and convert the corresponding data to a binary format
        for i in np.nditer(samp_idx):

            ppg = np.array(ppg_h5[i,:])

            if weights_SBP is not None and weights_DBP is not None:
                weight_SBP = weights_SBP[i]
                weight_DBP = weights_DBP[i]
            else:
                weight_SBP = 1
                weight_DBP = 1

            target = np.array(BP[i,:], dtype=np.float32)
            sub_idx = np.array(subject_idx[i])

            # create a dictionary containing the serialized data
            data = \
                {'ppg': _float_feature(ppg.tolist()),
                 'label': _float_feature(target.tolist()),
                 'subject_idx': _float_feature(sub_idx.tolist()),
                 'weight_SBP': _float_feature([weight_SBP]),
                 'weight_DBP': _float_feature([weight_DBP]),
                 'Nsamples': _float_feature([N_samples])}

            # write data to the .tfrecord target file
            feature = tf.train.Features(feature=data)
            example = tf.train.Example(features=feature)
            serialized = example.SerializeToString()

            writer.write(serialized)

        writer.close()


def ppg_hdf2tfrecord_sharded(h5_file, samp_idx, tfrecordpath, Nsamp_per_shard, modus='train', weights_SBP=None, weights_DBP=None):
    '''
    Save PPG/BP pairs as .tfrecord files. Save defined number os samples per file (Sharding)
    Weights can be defined for each sample.
    :param h5_file: File that contains the whole dataset (in .h5 format), created by
    :param samp_idx: sample indizes from the dataset in the h5. file that are used to create this tfrecords dataset
    :param tfrecordpath: full path for storing the .tfrecord files
    :param Nsamp_per_shard: number of samples per shard/.tfrecord file
    :param modus: define if the data is stored in the "train", "val" or "test" subfolder of "tfrecordpath"
    :param weights_SBP: sample weights for the systolic BP (optional)
    :param weights_DBP: sample weights for the diastolic BP (optional)
    :return:
    '''

    base_filename = join(tfrecordpath, 'MIMIC_III_ppg')
    N_samples = len(samp_idx)

    # calculate the number of Files/shards that are needed to stroe the whole dataset
    N_shards = np.ceil(N_samples / Nsamp_per_shard).astype(int)

    # iterate over every shard
    for i in range(N_shards):
        idx_start = i * Nsamp_per_shard
        idx_stop = (i + 1) * Nsamp_per_shard
        if idx_stop > N_samples:
            idx_stop = N_samples

        idx_curr = samp_idx[idx_start:idx_stop]
        output_filename = '{0}_{1}_{2:05d}_of_{3:05d}.tfrecord'.format(base_filename,
                                                                       modus,
                                                                       i + 1,
                                                                       N_shards)
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(dt_string, ': processing ',
              modus,
              ' shard ', str(i + 1), ' of ', str(N_shards))
        ppg_hdf2tfrecord(h5_file, output_filename, idx_curr, weights_SBP=weights_SBP, weights_DBP=weights_DBP)


def h5_to_tfrecords(SourceFile, tfrecordsPath, N_train=9e5, N_val=3e5, N_test=3e5,
                    save_tfrecords=True, split_strategy='r', enlarge_ratio=1.0, random_seed=0):
    '''
    Split the collected data into training, validation and test set based on 'split_strategy', and then
    save the splitted dataset into tfrecords (binary) format for further training.
    :param SourceFile:
    :param tfrecordsPath:
    :param N_train:
    :param N_val:
    :param N_test:
    :param save_tfrecords:
    :param split_strategy: splitting strategy that used for generating training, validation and test test from a data collected from multiple subject/records.
    :param enlarge_ratio:
    :param random_seed:
    :return:
    '''
    N_train = int(N_train)
    N_val = int(N_val)
    N_test = int(N_test)

    if enlarge_ratio == 1.0:
        split_strategy_ext = split_strategy
    else:
        split_strategy_ext = split_strategy +'_' + str(enlarge_ratio)

    tfrecord_path_train = join(tfrecordsPath, split_strategy_ext, str(random_seed), 'train')
    print('exists(tfrecord_path_train): {}'.format(exists(tfrecord_path_train)))
    print('isdir(tfrecord_path_train): {}'.format(isdir(tfrecord_path_train)))

    if not exists(tfrecord_path_train) or not isdir(tfrecord_path_train):
        makedirs(tfrecord_path_train, exist_ok=True)

    tfrecord_path_val = join(tfrecordsPath, split_strategy_ext, str(random_seed), 'val')
    if not exists(tfrecord_path_val) or not isdir(tfrecord_path_val):
        makedirs(tfrecord_path_val, exist_ok=True)

    tfrecord_path_test = join(tfrecordsPath, split_strategy_ext, str(random_seed), 'test')
    if not exists(tfrecord_path_test) or not isdir(tfrecord_path_test):
        makedirs(tfrecord_path_test, exist_ok=True)

    csv_path = join(tfrecordsPath, split_strategy_ext, str(random_seed))
    fig_path = join(tfrecordsPath, split_strategy_ext, str(random_seed))

    Nsamp_per_shard = 1000 # controls the capacity of each file

    print('Source file: {}'.format(SourceFile))

    with h5py.File(SourceFile, 'r') as f:
        BP = np.array(f.get('/label'))
        BP = np.round(BP)
        BP = np.transpose(BP)
        subject_idx = np.squeeze(np.array(f.get('/subject_idx')))

    N_samp_total = BP.shape[1]
    subject_idx = subject_idx[:N_samp_total]

    # Randomly select 750 subjects from the dataset composed of over 4500 subjects.
    subject_labels, SampSubject_hist = np.unique(subject_idx, return_counts=True)
    cumsum_samp = np.cumsum(SampSubject_hist)

    print('np.nonzero(cumsum_samp>(N_train+N_val+N_test))[0][0]: {}'.format(
        np.nonzero(cumsum_samp > (N_train + N_val + N_test))[0][0]))

    '''
    enlarge_ratio is set to 1.0 by default, 750 records used.
                                       2.0 -> 1500 records used.
                                       0.5 -> 375 records used.
    '''
    num_total_samples = (N_train + N_val + N_test) * enlarge_ratio
    N_train = int(N_train * enlarge_ratio)
    N_val = int(N_val * enlarge_ratio)
    N_test = int(N_test * enlarge_ratio)
    print('Enlarge_ratio: {}, there are totally {} samples.'.format(enlarge_ratio, num_total_samples))

    subject_labels_sel = subject_labels[: np.nonzero(cumsum_samp > num_total_samples)[0][0]]
    SampSubject_hist_sel = SampSubject_hist[: np.nonzero(cumsum_samp > num_total_samples)[0][0]]

    # Divide the dataset into training, validation and test set
    # -------------------------------------------------------------------------------
    if split_strategy in ['r', 'ru']:
        valid_idx = np.arange(subject_idx.shape[-1])

        # divide the subjects into training, validation and test subjects
        subjects_train_labels, subjects_val_labels = train_test_split(subject_labels_sel, test_size=0.4, random_state=random_seed)
        subjects_val_labels, subjects_test_labels = train_test_split(subjects_val_labels, test_size=0.5, random_state=random_seed)

        # Calculate samples belonging to training, validation and test subjects
        train_part = valid_idx[np.isin(subject_idx, subjects_train_labels)]
        val_part = valid_idx[np.isin(subject_idx, subjects_val_labels)]
        test_part = valid_idx[np.isin(subject_idx, subjects_test_labels)]

        # draw a number samples defined by N_train, N_val and N_test from the training, validation and test subjects
        if split_strategy == 'r':
            idx_train = np.random.choice(train_part, N_train, replace=False)
            idx_val = np.random.choice(val_part, N_val, replace=False)
            idx_test = np.random.choice(test_part, N_test, replace=False)
        else:
            assert (split_strategy == 'ru')
            idx_train = np.random.choice(train_part, 1, replace=False)
            idx_val = np.random.choice(val_part, 1, replace=False)
            idx_test = np.random.choice(test_part, 1, replace=False)
    else:
        # Create a subset of the whole dataset by drawing a number of subjects from the dataset. The total number of
        # samples contributed by those subjects must equal N_train + N_val + _N_test
        assert(split_strategy in ['s', 'si', 'sir'])

        if split_strategy == 's':
            idx_valid = np.nonzero(np.isin(subject_idx, subject_labels_sel))[0]  # the total index of samples with subject idx belonging to subject_labels_train

            # divide subset randomly into training, validation and test set
            # idx_train, idx_val = train_test_split(idx_valid, train_size=N_train, test_size=N_val + N_test,random_state=random_seed)
            idx_train, idx_val = train_test_split(idx_valid, train_size=0.6, test_size=0.4, random_state=random_seed)
            idx_val, idx_test = train_test_split(idx_val, test_size=0.5, random_state=random_seed)
        else:
            num_subjects = subject_labels_sel.shape[0]
            idx_train, idx_val, idx_test = np.empty((0), dtype=np.int32), np.empty((0), dtype=np.int32), np.empty((0), dtype=np.int32)

            for sid in range(num_subjects):
                idx_valid = np.nonzero(np.isin(subject_idx, subject_labels_sel[sid]))[0]

                if split_strategy == 'sir':
                    idx_train_tmp, idx_val_tmp = train_test_split(idx_valid, train_size=int(SampSubject_hist_sel[sid] * 0.6),
                                                                  test_size=int(SampSubject_hist_sel[sid] * 0.4), random_state=random_seed)
                    idx_val_tmp, idx_test_tmp = train_test_split(idx_val_tmp, test_size=0.5, random_state=random_seed)
                else:
                    assert (split_strategy == 'si')
                    idx_train_tmp = idx_valid[:int(SampSubject_hist_sel[sid] * 0.6)]
                    idx_val_tmp = idx_valid[int(SampSubject_hist_sel[sid] * 0.6):int(SampSubject_hist_sel[sid] * 0.8)]
                    idx_test_tmp = idx_valid[int(SampSubject_hist_sel[sid] * 0.8):]

                idx_train = np.hstack((idx_train, idx_train_tmp))
                idx_val = np.hstack((idx_val, idx_val_tmp))
                idx_test = np.hstack((idx_test, idx_test_tmp))

    # cmpt BP statistics
    idx_valid = np.nonzero(np.isin(subject_idx, subject_labels_sel))[0]
    SBP_total = BP[0, idx_valid]
    DBP_total = BP[1, idx_valid]
    print('Total statistics, \n SBP, Range: {}-{}, Mean: {}, Std: {}'.
          format(min(SBP_total), max(SBP_total), np.mean(SBP_total), np.std(SBP_total)))
    print('DBP, Range: {}-{}, Mean: {}, Std: {}'.
          format(min(DBP_total), max(DBP_total), np.mean(DBP_total), np.std(DBP_total)))

    # plot BP dynamics
    sbp_by_individual = {}
    dbp_by_individual = {}

    for subject_label_t in subject_labels_sel:
        idxOfSubject = np.nonzero(np.isin(subject_idx, [subject_label_t]))[0]

        sbp_by_individual[subject_label_t] = BP[0, idxOfSubject].tolist()
        dbp_by_individual[subject_label_t] = BP[1, idxOfSubject].tolist()

    plot_BP_dynamics([sbp_by_individual, dbp_by_individual], labels=['SBP', 'DBP'], fig_path=fig_path)

    # save ground truth BP values of training, validation and test set in csv-files for future reference
    BP_train = BP[:, idx_train]
    d = {"SBP": np.transpose(BP_train[0, :]), "DBP": np.transpose(BP_train[1, :])}
    train_set = pd.DataFrame(d)
    train_set.to_csv(join(csv_path, 'MIMIC-III_BP_trainset.csv'))

    BP_val = BP[:, idx_val]
    d = {"SBP": np.transpose(BP_val[0, :]), "DBP": np.transpose(BP_val[1, :])}
    train_set = pd.DataFrame(d)
    train_set.to_csv(join(csv_path, 'MIMIC-III_BP_valset.csv'))

    BP_test = BP[:, idx_test]
    d = {"SBP": np.transpose(BP_test[0, :]), "DBP": np.transpose(BP_test[1, :])}
    train_set = pd.DataFrame(d)
    train_set.to_csv(join(csv_path, 'MIMIC-III_BP_testset.csv'))

    # save fig (BP_distribution)
    fig_name = join(fig_path, 'MIMIC-III_BP_distribution')
    plot_BP_distribution(BP_train, BP_val, BP_test, fig_name=fig_name)

    # create tfrecord dataset
    # ----------------------------
    if save_tfrecords:
        np.random.shuffle(idx_train)
        ppg_hdf2tfrecord_sharded(SourceFile, idx_test, tfrecord_path_test, Nsamp_per_shard, modus='test', split_strategy=split_strategy)
        ppg_hdf2tfrecord_sharded(SourceFile, idx_train, tfrecord_path_train, Nsamp_per_shard, modus='train', split_strategy=split_strategy)
        ppg_hdf2tfrecord_sharded(SourceFile, idx_val, tfrecord_path_val, Nsamp_per_shard, modus='val',split_strategy=split_strategy)
    print("Script finished")


def plot_BP_dynamics(bp_dynamics, labels=['SBP', 'DBP'], fig_path='./results'):
    '''
    Visualize individual BPs (SBP and DBP) dynamics.
    :param bp_dynamics: [dict, dict,], with record idx served as key, the belonging BP value list as value
    :param labels:
    :param fig_path:
    :return:
    '''
    for bp_dynamic, label in zip(bp_dynamics, labels):

        # sort according to the ascending order of the BP range limits (i.e BPRange_{max}-BPRange_{minx})
        bp_dynamic = dict(sorted(bp_dynamic.items(), key=lambda x: max(x[1]) - min(x[1]), reverse=False))  # 按字典集合中，每一个元组的第二个元素排列。

        bp_individual_dynamic = [max(value) - min(value) for key, value in bp_dynamic.items()]

        # cmpt individual BP dynamics statics
        print('Individual dynamics\n, {}, Range: {}-{}, Mean: {}, Std: {}'
              .format(label, min(bp_individual_dynamic), max(bp_individual_dynamic),
               np.mean(bp_individual_dynamic), np.std(bp_individual_dynamic)))

        xlabel_list = []
        bp_dynamic_list = []
        for key in bp_dynamic.keys():
            xlabel_list.append(key)
            bp_dynamic_list.append(bp_dynamic[key])

        fig, ax = plt.subplots(1, 1, figsize=(40, 8))
        box1 = ax.boxplot(x=bp_dynamic_list, labels=xlabel_list, showmeans=True, notch=False, patch_artist=True,
                          whis=1.5, widths=0.5, vert=True, medianprops={'color': 'red'}, boxprops=dict(color='black'),
                          whiskerprops={'color': 'black'}, capprops={'color': 'darkmagenta'}, flierprops={'color': 'black',
                                                                                                          'markeredgecolor': 'black'})
        ax.set_xticklabels(xlabel_list, rotation=90, fontsize=5)
        ax.set_xlabel('subject/record', fontsize=15, family='Palatino')
        ax.set_ylabel('BP distribution', fontsize=15, family='Palatino')
        ax.grid(b=True, which='major', axis='y', color='k', linestyle='--', linewidth=0.5)

        ax_right = ax.twinx()
        ax_right.plot(bp_individual_dynamic, color='b', marker='x')
        ax_right.set_ylabel('Individual dynamics', fontsize=15, family='Palatino')

        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        fig_name = 'individual_{}_dynamics'.format(label)
        plt.savefig(os.path.join(fig_path, fig_name + '.svg'), format='svg')
        plt.savefig(os.path.join(fig_path, fig_name + '.jpg'), format='jpg')

        # plot the distribution of individual BP dynamics
        fig, ax2 = plt.subplots(1, 1, figsize=(8, 8))
        alpha = 0.7
        rwidth = 0.85
        ax2.hist(x=bp_individual_dynamic, bins=20, alpha=alpha, rwidth=rwidth, color='r')
        ax2.set_xlabel('Individual BP difference(mmHg)', fontsize=15)
        ax2.set_ylabel('Frequency', fontsize=15)

        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        fig_name = 'individual_{}_dynamics2'.format(label)
        plt.savefig(os.path.join(fig_path, fig_name + '.svg'), format='svg')
        plt.savefig(os.path.join(fig_path, fig_name + '.jpg'), format='jpg')

        plt.close()


def plot_BP_distribution(BP_train, BP_val, BP_test, fig_name=None):
    '''
    Visualize the distribution of BPs (SBP and DBP) in the splitted training, validation and test set.
    :param BP_train: size: n x 2
    :param BP_val:
    :param BP_test:
    :return:
    '''
    num_bins = 30
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

    ax1.hist(BP_train[0, :], bins=num_bins, label='SBP')
    ax1.hist(BP_train[1, :], bins=num_bins, label='DBP')
    ax1.set_xlabel('Train BP range')
    ax1.set_ylabel('Frequency')
    ax1.legend()

    ax2.hist(BP_val[0, :], bins=num_bins, label='SBP')
    ax2.hist(BP_val[1, :], bins=num_bins, label='DBP')
    ax2.set_xlabel('Validation BP range')
    ax2.set_ylabel('Frequency')
    ax2.legend()

    ax3.hist(BP_test[0, :], bins=num_bins, label='SBP')
    ax3.hist(BP_test[1, :], bins=num_bins, label='DBP')
    ax3.set_xlabel('Test BP range')
    ax3.set_ylabel('Frequency')

    ax3.legend()

    if fig_name is not None:
        plt.savefig(fig_name + '.jpg', format='jpg')
        plt.savefig(fig_name + '.svg', forma='svg')


if __name__ == "__main__":
    np.random.seed(seed=42)

    if len(argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument('input', type=str, help="Path to the .h5 file containing the dataset")
        parser.add_argument('output', type=str, help="Target folder for the .tfrecord files")
        parser.add_argument('--ntrain', type=int, default=1e6,
                            help="Number of samples in the training set (default: 1e6)")
        parser.add_argument('--nval', type=int, default=2.5e5,
                            help="Number of samples in the validation set (default: 2.5e5)")
        parser.add_argument('--ntest', type=int, default=2.5e5,
                            help="Number of samples in the test set (default: 2.5e5)")

        parser.add_argument('--enlarge_ratio', type=float, default=1.0, help='')

        parser.add_argument('--random_seed', type=int, default=0,
                            help='')
        parser.add_argument('--split_strategy', type=str, default='r',
                            help='') # 'r': record level/ 'ru': record level with single saple,
        # 'sa': sample level with aggregated operation/ 'si': sample level with intra record split, 'sir': sample level with intra record random split

        args = parser.parse_args()
        SourceFile = args.input
        tfrecordsPath = args.output
        split_strategy = args.split_strategy
        enlarge_ratio = args.enlarge_ratio
        random_seed = args.random_seed
    else:
        HomePath = expanduser("~")
        SourceFile = join(HomePath, 'data', 'MIMIC-III_BP', 'MIMIC-III_ppg_dataset_org.h5')
        tfrecordsPath = join(HomePath, 'test')

    # used for debug in the commond line
    #import pdb
    #pdb.set_trace()

    h5_to_tfrecords(SourceFile=SourceFile, tfrecordsPath=tfrecordsPath, save_tfrecords=True,
                    split_strategy=split_strategy, enlarge_ratio=enlarge_ratio, random_seed=random_seed)
