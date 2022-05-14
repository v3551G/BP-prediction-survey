#! /usr/bin/env
# -*- coding:utf-8 -*-

import numpy as np
import scipy.stats as sta
import scikit_posthocs as sp

from analyze.significant_test.tool.friedman_test import cmpt_average_rank
from analyze.significant_test.tool.visualize_nimenyi_test import visualize_posthoc_nemenyi_test, visualize_posthoc_nemenyi_test2

'''
    Analyzing whether the difference of the results based on different splitting strategy is significant.
    masterqkk@outlook.com
'''

if __name__ == '__main__':
    '''
    N = 5, k = 4 (r/s/si/sir)
    '''
    data_mae_sbp = [[12.093, 8.077, 8.494, 8.363],
                    [12.436, 8.207, 8.303, 8.386],
                    [12.363, 8.195, 8.335, 8.447],
                    [11.978, 8.291, 8.407, 8.474],
                    [12.571, 8.172, 8.478, 8.350]]
    data_mae_dbp = [[6.529, 4.665, 4.898, 4.822],
                    [6.665, 4.738, 4.770, 4.844],
                    [6.716, 4.695, 4.800, 4.825],
                    [6.315, 4.774, 4.856, 4.831],
                    [6.631, 4.687, 4.930, 4.828]]

    data_mae_sbp = np.array(data_mae_sbp, dtype=np.float32)
    data_mae_dbp = np.array(data_mae_dbp, dtype=np.float32)

    # cmpt average rank
    rank_sbp = cmpt_average_rank(data_mae_sbp)
    rank_dbp = cmpt_average_rank(data_mae_dbp)

    print('Average rank, SBP: {} \n DBP: {}'.format(rank_sbp, rank_dbp))

    # friedman test
    stats_sbp, pvalue_sbp = sta.friedmanchisquare(data_mae_sbp[:, 0], data_mae_sbp[:, 1], data_mae_sbp[:, 2], data_mae_sbp[:, 3])
    stats_dbp, pvalue_dbp = sta.friedmanchisquare(data_mae_dbp[:, 0], data_mae_dbp[:, 1], data_mae_dbp[:, 2], data_mae_dbp[:, 3])

    print('Friedman test, \n SBP, stats: {}, pvalue: {} \n stats: {}, pvalue: {}'.format(stats_sbp, pvalue_sbp, stats_dbp, pvalue_dbp))

    '''
        posthoc-nemenyi test, q_{alpha=0.05}^{k=4} = 2.569, N=5, k=4
        CD = q_{alpha} * \sqrt{k(k+1)/(6N)} = 2.0975797164033283
        the output of 'posthoc_nemenyi_friedman' is a matrix consists of  ARD value of paired comparison, 
           if the ARD value is smaller than \alpha, then reject null  hypothesis, i.e the performance is statistically significant.
        
    '''
    pvalue_phnm_sbp = sp.posthoc_nemenyi_friedman(data_mae_sbp)
    pvalue_phnm_dbp = sp.posthoc_nemenyi_friedman(data_mae_dbp)

    print('Posthoc nimenyi test, \n SBP: {}, \n DBP: {}'.format(pvalue_phnm_sbp, pvalue_phnm_dbp))

    # visualize Posthoc nimenyi test results
    methods_with_score_sbp = {
        'ResNet model based on ' + r'$\textcircled{1}$': rank_sbp[0],
        'ResNet model based on ' + r'$\textcircled{3}$': rank_sbp[1],
        'ResNet model based on ' + r'$\textcircled{4}$': rank_sbp[2],
        'ResNet model based on ' + r'$\textcircled{5}$': rank_sbp[3]
    }
    methods_with_score_dbp = {
        'ResNet model based on ' + r'$\textcircled{1}$': rank_dbp[0],
        'ResNet model based on ' + r'$\textcircled{3}$': rank_dbp[1],
        'ResNet model based on ' + r'$\textcircled{4}$': rank_dbp[2],
        'ResNet model based on ' + r'$\textcircled{5}$': rank_dbp[3]
    }

    #visualize_posthoc_nemenyi_test(methods_with_score_sbp, critical_lines=[[0, 2], [1, 3]], fig_name='nimenyi_test_{}.svg'.format('SBP'))
    #visualize_posthoc_nemenyi_test(methods_with_score_dbp, critical_lines=[[0, 2], [1, 3]], fig_name='nimenyi_test_{}.svg'.format('DBP'))

    num_exp = data_mae_sbp.shape[0]
    visualize_posthoc_nemenyi_test2(methods_with_score_sbp, num_exp, alpha=0.05, filename='nimenyi_test_{}.svg'.format('SBP'))
    visualize_posthoc_nemenyi_test2(methods_with_score_dbp, num_exp, alpha=0.05, filename='nimenyi_test_{}.svg'.format('DBP'))

