#! /usr/bin/env
# -*- coding:utf-8 -*-

import os
import csv
import pandas as pd
import matplotlib.pyplot as plt

'''
    Results analyzing tools,
    masterqkk@outlook.com,
'''

base_dir = '../results'

split_strategy = 'r'
file_path = os.path.join(base_dir, split_strategy)
files = os.listdir(file_path)
files = [x for x in files if x.endswith('.csv') and 'learningcurve' in x]

df = pd.read_csv(os.path.join(file_path, files[0]))

DBP_loss = df['DBP_loss'].values
SBP_loss = df['SBP_loss'].values
val_SBP_loss = df['val_SBP_loss'].values
val_DBP_loss = df['val_DBP_loss'].values

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.plot(SBP_loss, label=r'$Loss_{SBP}$')
ax.plot(DBP_loss, label=r'$Loss_{DBP}$')
ax.plot(val_SBP_loss, label=r'$Loss_{SBP} val$')
ax.plot(val_DBP_loss, label=r'$Loss_{DBP} val$')

ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.legend()

fig_name = '{}-{}-{}'.format(files[0].split('/')[-1].split('.')[0], split_strategy, 'loss')
plt.savefig(os.path.join('.', fig_name + '.jpg'), format='jpg')
plt.savefig(os.path.join('.', fig_name + '.svg'), format='svg')

#
DBP_mae = df['DBP_mae'].values
SBP_mae = df['SBP_mae'].values
val_SBP_mae = df['val_SBP_mae'].values
val_DBP_mae = df['val_DBP_mae'].values

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.plot(SBP_mae, label=r'$MAE_{SBP}$')
ax.plot(DBP_mae, label=r'$MAE_{DBP}$')
ax.plot(val_SBP_mae, label=r'$MAE_{SBP} val$')
ax.plot(val_DBP_mae, label=r'$MAE_{DBP} val$')

ax.set_xlabel('Epochs')
ax.set_ylabel('MAE')
ax.legend()

fig_name = '{}-{}-{}'.format(files[0].split('/')[-1].split('.')[0], split_strategy, 'mae')
plt.savefig(os.path.join('.', fig_name + '.jpg'), format='jpg')
plt.savefig(os.path.join('.', fig_name + '.svg'), format='svg')

