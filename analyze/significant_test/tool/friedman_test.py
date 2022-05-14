#! /usr/bin/env
# -*- coding:utf-8 -*-
import numpy as np
import scipy.stats as sta
import scikit_posthocs as sp

'''
  A demo for Friedman test
  masterqkk@outlook.com
'''


def Friedman(n, k, data_matrix):
    '''
    Friedman检验, not complete.
    参数：数据集个数n, 算法种数k, 排序矩阵data_matrix: ranked_matrix
    返回值是Tf
    '''

    # 计算每个算法的平均序值，即求每一列的排序均值
    hang, lie = data_matrix.shape  # 查看数据形状
    XuZhi_mean = list()
    for i in range(lie):  # 计算平均序值
        XuZhi_mean.append(data_matrix[:, i].mean())

    sum_mean = np.array(XuZhi_mean)  # 转换数据结构方便下面运算
    ## 计算总的排序和即西伽马ri^2
    sum_ri2_mean = (sum_mean ** 2).sum()
    # 计算Tf
    result_Tx2 = (12 * n) * (sum_ri2_mean - ((k * (k + 1) ** 2) / 4)) / (k * (k + 1))
    result_Tf = (n - 1) * result_Tx2 / (n * (k - 1) - result_Tx2)

    return result_Tf


def cmpt_average_rank(data):
    '''
    :param data: [N, K] the results of K methods on N datasets/experiments.
    :return:
    '''
    row, column = data.shape[0], data.shape[1]
    rank_matrix = np.zeros(shape=(row, column), dtype=np.float32)
    for idx in range(row):
        print(data[idx, :])
        x = np.argsort(data[idx, :])
        '''
        d[np.argsort(d)] = np.sort(d),
        np.argsort(np.argsort(d)) = 
        '''
        rank_matrix[idx, :] = np.argsort(np.argsort(data[idx, :])) + 1 # cmpt the rank of each method.

        #TODO, ready for debugging.
        '''
        # adjust if there are same value in each row
        raw_value_ranked = np.sort(data[idx, :])
        iter = True
        sp = 0
        while iter:
            cp = sp+1
            exist_rep = False
            value_exist = -1
            for i in range(cp, column):
                if raw_value_ranked[i] == raw_value_ranked[sp]:
                    exist_rep = True
                    value_exist = raw_value_ranked[sp]
                    cp = i
                    continue
                else:
                    sp = cp
                    break
            #
            if exist_rep:
                sel = np.where(data[idx, :] == value_exist)
                rank_matrix[idx, sel] = np.mean(rank_matrix[idx, sel])
        '''
    average_rank = np.mean(rank_matrix, axis=0)
    return average_rank


if __name__ == '__main__':
    data = np.array([[1, 2, 3],
                     [1, 2.5, 2.5],
                     [1, 2, 3],
                     [1, 2, 3]]
                    )  # 书上的数据
    #Tf = Friedman(4, 3, data)
    #print(Tf)

    '''
    Friedman test:
       Null hypothesis, H0: all methods are comparable
       pvalue = 0.02 < alpha=0.05, reject H0, -> exist difference
    '''
    stats, pvalue = sta.friedmanchisquare([1,1,1,1], [2,2.5,2,2], [3,2.5,3,3])
    print('stats: {}, \n pvalue; {}'.format(stats, pvalue))



    '''
    Posthoc nemenyi test:
        input: [N,k] the matrix of performance of k methods on N datasets
        
        after obtaining the pvalue_matrix, then compare each element with CD=q_{alpha} * \sqrt{k(k+1)/(6N)}, q_{alpha} is obtained from Table.
               if ARD(alg1, alg2) > CD, reject H0, exist significant differences
    '''
    pvalue_ph = sp.posthoc_nemenyi_friedman(data)
    print('posthoc nemenyi test: \n {}'.format(pvalue_ph))
