#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt

import bibtexparser as bp
from bibtexparser.bparser import BibTexParser
from bibtexparser.customization import *  # convert_to_unicode, etc

'''
  Analyizing tools of *.bib format file, used to calculate the number of published paper over years,
  masterqkk@outlook.com, 20220215.
'''


def read_file(file_name):
    with open(file_name, encoding='utf-8') as bibfile:
        parser = BibTexParser()
        #parser.customization = convert_to_unicode
        parser.customization = customization

        bibdata = bp.load(bibfile, parser=parser)

        return bibdata


def customization(record):
    record = type(record)
    record = author(record)
    record = editor(record)
    record = journal(record)
    record = keyword(record)
    record = link(record)
    record = page_double_hyphen(record)
    record = doi(record)

    return record


def statistic(dict_list):
    '''
    :param dict_list: a list of references
    :return:
    '''
    publish_res = {'noname': 0}
    survey_res = {}
    total_res = {}
    ptt_res = {}
    dl_res = {}

    num_ref = len(dict_list)

    for id in range(num_ref):
        ref = dict_list[id]

        if not 'year' in ref.keys():
            continue

        year  = ref['year']
        title = ref['title'].lower()

        if 'blood pressure' in title and ('prediction' in title or 'predicting' in title or 'predictive' in title or 'estimating' in title
                                          or 'estimation' in title or 'estimator' in title or 'estimate' in title or 'measurement' in title
                                          or 'measuring' in title or 'measure' in title or 'monitoring' in title or 'monitor' in title
                                          or 'tracking' in title or 'sensing' in title or 'inference' in title or 'detection' in title
                                          or 'assessment' in title or 'assessing' in title or 'generative' in title or 'imputation' in title
                                          or 'reconstruction' in title or 'translating' in title or 'indicator' in title or 'classification' in title
                                          or title in ['the machine learnings leading the cuffless ppg blood pressure sensors into the next stage']
                                        ):
            #

            if 'journal' in ref.keys():
                publish_source = ref['journal']['name']
            elif 'booktitle' in ref.keys():
                publish_source = ref['booktitle'].split('(')[1].split(')')[0]
            else:
                publish_source = 'noname'

            if publish_source in publish_res.keys():
                publish_res[publish_source] = publish_res[publish_source] + 1
            else:
                publish_res[publish_source] = 1

            #
            if 'survey' in title or 'review' in title  or 'proposal' in title or title in survey_special_titles:
                if year in survey_res.keys():
                    survey_res[year] = survey_res[year] + 1
                else:
                    survey_res[year] = 1

            else:
                if year in total_res.keys():
                    total_res[year] = total_res[year] + 1
                else:
                    total_res[year] = 1

                if 'deep' in title or 'neural' in title or 'network' in title or 'rnn' in title or 'cnn' in title or 'u-net' in title\
                        or 'lstm' in title or title == 'estimation of the blood pressure waveform using electrocardiography':
                    if year in dl_res.keys():
                        dl_res[year] = dl_res[year] + 1
                    else:
                        dl_res[year] = 1
                elif 'pulse transit time' in title or 'PTT' in title:
                    if year in ptt_res.keys():
                        ptt_res[year] = ptt_res[year] + 1
                    else:
                        ptt_res[year] = 1

    # pad keys for survey_res and ptt_res
    key_full_set = list(total_res.keys())
    key_survey = list(survey_res.keys())
    key_ptt = list(ptt_res.keys())

    print(key_full_set)

    for key in key_full_set:
        if not key in key_survey:
            survey_res[key] = 0
        if not key in key_ptt:
            ptt_res[key] = 0

    # sort according to key values, i.e year ascending.
    survey_res = dict(sorted(zip(survey_res.keys(), survey_res.values())))
    total_res = dict(sorted(zip(total_res.keys(), total_res.values())))
    dl_res = dict(sorted(zip(dl_res.keys(), dl_res.values())))
    ptt_res = dict(sorted(zip(ptt_res.keys(), ptt_res.values())))

    return survey_res, total_res, dl_res, ptt_res, publish_res


def plot_trend(data_list, name_list):
    num_variable = len(data_list)
    fig = plt.figure(figsize=(10, 5))
    for id in range(num_variable):
        plt.bar(data_list[id].keys(), data_list[id].values(), width=2, color=colors[id], label=name_list[id])

    plt.legend(fontsize=16)
    plt.xlabel('Year')
    plt.ylabel('# papers')
    #plt.show()
    plt.savefig(os.path.join(result_path, 'publish_trend.jpg'))
    plt.savefig(os.path.join(result_path, 'publish_trend.svg'), format='svg')


def plot_trend2(data_list, name_list):
    '''
    Visualize the trend in terms of number of publications along years.
    :param data_list:
    :param name_list:
    :return:
    '''
    num_variable = len(data_list)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for id in range(1, num_variable):
        ax.bar(data_list[id].keys(), data_list[id].values(), width=2, color=colors[id], label=name_list[id])

    ax.legend(fontsize=16)
    ax.set_xlabel('Year')
    ax.set_ylabel('# papers')


    ax_right = ax.twinx()
    ax_right.plot(data_list[0].values(),color=colors[0], label=name_list[0], marker='*', markersize=8)
    ax_right.set_ylim([0, 20])

    ax_right.set_ylabel('# papers')
    ax_right.legend(fontsize=16, loc=3)


    #plt.show()
    plt.savefig(os.path.join(result_path, 'publish_trend2.jpg'))
    plt.savefig(os.path.join(result_path, 'publish_trend2.svg'), format='svg')


def plot_pie(data, truct_num=20):
    '''
    Visualize the distribution of published journal or conferences.
    masterqkk, 20200505
    :param data: dict format
    :return:
    '''
    # truncate data , the first 20 items
    num_total_publish = np.sum(list(data.values()))
    num_total_journal_conf = len(list(data.values()))
    selected_publish_source = sorted(data, key=lambda x: data[x], reverse=True)[:truct_num]
    print('data_sorted: {}'.format(selected_publish_source))
    selected_data = {}
    acc_num_publish = 0
    for name in selected_publish_source:
        selected_data[name] = data[name]
        acc_num_publish += data[name]
    #
    selected_data['Others ({} publish sources)'.format(str(num_total_journal_conf - truct_num))] = num_total_publish - acc_num_publish

    labels = list(selected_data.keys())
    values = list(selected_data.values())
    expodes = tuple([0.2] * len(labels))

    #colors = ['red','orange','yellow','green','purple','blue','black']

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.pie(values, autopct='%.2f%%', explode=expodes, labels=labels, pctdistance=0.8, labeldistance=1.1, shadow=True)

    plt.savefig(os.path.join(result_path, 'publish_journal_proportion2.jpg'))
    plt.savefig(os.path.join(result_path, 'publish_journal_proportion2.svg'), format='svg')


if __name__ == "__main__":
    file_path = '../'
    result_path = './result'
    file_name = 'mybibfile.bib'

    survey_special_titles = ['cuffless single-site photoplethysmography for blood pressure monitoring',
                             'the machine learnings leading the cuffless ppg blood pressure sensors into the next stage',
                             'blood pressure measurement: clinic, home, ambulatory, and beyond',
                             'oscillometric blood pressure estimation: past, present, and future',
                             'cuffless blood pressure monitors: principles, standards and approval for medical use'
                             ]

    ptt_special_titles = ['blood pressure estimation using on-body continuous wave radar and photoplethysmogram in various posture and exercise conditions',
                          ]

    colors = ['sandybrown', 'springgreen', 'orangered', 'royalblue', 'r', 'y', 'mediumblue']

    bibdata = read_file(os.path.join(file_path, file_name))
    file_list = bibdata.entries
    # print('{} \n {}'.format(file_list[0]['author'], file_list[0]['title']))

    survey_res, total_res, dl_res, ptt_res, publish_res = statistic(file_list)
    print('survey: {}, \n total: {}, \n deep learning methods: {}, \n ptt methods: {}'.format(survey_res, total_res, dl_res, ptt_res))

    plot_trend2(data_list=[survey_res, total_res, dl_res, ptt_res], name_list=['Survey', 'Total', 'Deep learning methods', 'PTT methods'])
    #
    print('published journal/confernece: {}'.format(publish_res))
    plot_pie(publish_res)






