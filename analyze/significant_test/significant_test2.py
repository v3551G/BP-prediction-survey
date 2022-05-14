#! /usr/bin/env
# -*- coding:utf-8 -*-

import numpy as np
import scipy.stats as sta
import scikit_posthocs as sp

from analyze.significant_test.tool.friedman_test import cmpt_average_rank
from analyze.significant_test.tool.visualize_nimenyi_test import visualize_posthoc_nemenyi_test, visualize_posthoc_nemenyi_test2

'''
    Analyzing whether the difference of the results of models trained using different size of dataset.
    masterqkk@outlook.com, 20220402.
'''

if __name__ == '__main__':
    '''
    N = 5, k = 4 (r0.5/r1.0/r2.0/r4.0)
    '''
    data_mae_sbp = [[11.309, 12.093, 13.027, 15.141, 14.310],
                    [12.468, 12.436, 13.234, 14.725, 14.568],
                    [11.203, 12.363, 12.710, 14.922, 15.476],
                    [12.017, 11.978, 12.931, 14.713, 14.651],
                    [11.791, 12.571, 12.014, 14.549, 14.473]]
    data_mae_dbp = [[6.418, 6.529, 7.316, 8.169, 7.772],
                    [6.418, 6.665, 7.008, 7.895, 7.907],
                    [6.518, 6.716, 7.050, 8.034, 8.240],
                    [6.755, 6.315, 7.213, 8.488, 7.978],
                    [6.355, 6.631, 6.813, 7.757, 7.877]]

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
        'ResNet model (v1)': rank_sbp[0],
        'ResNet model (v2)': rank_sbp[1],
        'ResNet model (v3)': rank_sbp[2],
        'ResNet model (v4)': rank_sbp[3]
    }
    methods_with_score_dbp = {
        'ResNet model (v1)': rank_dbp[0],
        'ResNet model (v2)': rank_dbp[1],
        'ResNet model (v3)': rank_dbp[2],
        'ResNet model (v4)': rank_dbp[3]
    }

    #visualize_posthoc_nemenyi_test(methods_with_score_sbp, critical_lines=[[0, 2], [1, 3]], fig_name='nimenyi_test_{}.svg'.format('SBP'))
    #visualize_posthoc_nemenyi_test(methods_with_score_dbp, critical_lines=[[0, 2], [1, 3]], fig_name='nimenyi_test_{}.svg'.format('DBP'))

    num_exp = data_mae_sbp.shape[0]
    visualize_posthoc_nemenyi_test2(methods_with_score_sbp, num_exp, alpha=0.05, filename='nimenyi_test_{}_datasize_effect.svg'.format('SBP'))
    visualize_posthoc_nemenyi_test2(methods_with_score_dbp, num_exp, alpha=0.05, filename='nimenyi_test_{}_datasize_effect.svg'.format('DBP'))

