#! /usr/bin/env
# -*- coding:utf-8 -*-
import svgwrite
import numpy as np
import Orange
from Orange.evaluation import compute_CD, graph_ranks
import matplotlib
matplotlib.rcParams['text.usetex'] = True

matplotlib.rcParams['text.latex.preamble'] = [
    '\\usepackage{pifont}',
    r'AtBeginDocument{\begin{CKJ}{UTF8}{gbsn}}',
    r'AtEndDocument{\end{CJK}}',
]

'''
    Visualization tools for post-hoc Nimenyi test, 
    masterqkk@outlook.com, 20220329
'''


def visualize_posthoc_nemenyi_test(methods_with_score, critical_lines=[[1, 4], [2, 6]], fig_name='test.svg'):
    '''
    Something wrong.
    :param methods_with_score: dict format
    :return:
    '''

    method_count = len(methods_with_score)
    # well we have to compare more than one method
    assert (method_count > 1)

    sorted_methods = sorted(methods_with_score, key=lambda x: methods_with_score[x])

    font_height = 14
    margin_ltbr = [70, 4*font_height, font_height, font_height]
    main_line_color = svgwrite.rgb(10, 10, 16, '%')
    text_color = 'red'
    critical_line_color = 'orange'

    # interval between methods
    line_height = 1.5*font_height
    interval_size = font_height*1.5
    interval_height = font_height*0.33
    # assume that 100px = 100%
    size = interval_size * (method_count-1)

    number_of_critical_lines = len(critical_lines)
    critical_line_space = number_of_critical_lines * line_height
    item_space = 1.5*line_height

    #critical_distance_value = max([methods_with_score[sorted_methods[x[1]]] - methods_with_score[sorted_methods[x[0]]] for x in critical_lines]) / float(method_count)
    critical_distance_value = 2.569 * np.sqrt(4*5/(6*5))
    critical_distance = size * critical_distance_value

    print('critical distance value (CD): {}'.format(critical_distance_value))

    dwg = svgwrite.Drawing(fig_name, profile='tiny')

    for interval_line_number in range(method_count):
        x_i = int(interval_size * interval_line_number)
        rank = method_count - interval_line_number
        text_element = svgwrite.text.Text(
                str(rank),
                x=[margin_ltbr[0] + x_i],
                y=[margin_ltbr[1] + font_height], fill=text_color
        )
        text_element["text-anchor"] = "middle"
        dwg.add(text_element)

        dwg.add(svgwrite.shapes.Line(
                (x_i+margin_ltbr[0], 4+font_height+margin_ltbr[1]),
                (x_i+margin_ltbr[0], 4.5+font_height+interval_height+margin_ltbr[1]),
                stroke=main_line_color)
        )

        method_position = size * methods_with_score[sorted_methods[interval_line_number]]/float(method_count)
        method_vertical_end = (method_position+margin_ltbr[0], 4+font_height+margin_ltbr[1])
        anchor = "start"
        if (rank > np.ceil(0.5*method_count)):
            method_horizontal_end = 0+margin_ltbr[0]
            method_junction = critical_line_space+(rank-np.ceil(0.5*method_count))*item_space+margin_ltbr[1]
            anchor = "end"
        else:
            method_horizontal_end = size+margin_ltbr[0]
            method_junction = critical_line_space+rank*item_space+margin_ltbr[1]
            anchor = "start"

        dwg.add(svgwrite.shapes.Line(
                method_vertical_end,
                (method_vertical_end[0], method_junction),
                stroke=main_line_color)
        )

        dwg.add(svgwrite.shapes.Line(
                (method_horizontal_end, method_junction),
                (method_vertical_end[0], method_junction),
                stroke=main_line_color)
        )

        text_element = svgwrite.text.Text(
                str(sorted_methods[rank-1]),
                x = [method_horizontal_end],
                y = [method_junction], fill=text_color)
        text_element["text-anchor"] = anchor
        dwg.add(text_element)

    for i in range(len(critical_lines)):
        critical_line = critical_lines[i]
        method_position_left = 0 + margin_ltbr[0] + size * methods_with_score[sorted_methods[method_count - (critical_line[0] + 1)]] / float(method_count)
        method_position_right = 0 + margin_ltbr[0] + size * methods_with_score[sorted_methods[method_count - (critical_line[1] + 1)]] / float(method_count)

        dwg.add(svgwrite.shapes.Line(
                (method_position_left, 4+font_height+interval_height+margin_ltbr[1]+(i+.5)*line_height*.5),
                (method_position_right, 4+font_height+interval_height+margin_ltbr[1]+(i+.5)*line_height*.5),
                stroke=critical_line_color)
        )

        dwg.add(svgwrite.shapes.Line(
            (method_position_left, 4+font_height+interval_height+margin_ltbr[1]+(i+.25)*line_height*.5),
            (method_position_left, 4+font_height+interval_height+margin_ltbr[1]+(i+.75)*line_height*.5),
            stroke=critical_line_color)
        )

        dwg.add(svgwrite.shapes.Line(
            (method_position_right, 4+font_height+interval_height+margin_ltbr[1]+(i+.25)*line_height*.5),
            (method_position_right, 4+font_height+interval_height+margin_ltbr[1]+(i+.75)*line_height*.5),
            stroke=critical_line_color)
        )

    text_element = svgwrite.text.Text(
               'Critical distance ' + r'$CD_{0.05}$' + '={0:.5f}'.format(critical_distance_value),
                x=[0+margin_ltbr[0]],
                y=[2.5*font_height], fill=critical_line_color
    )

    dwg.add(text_element)

    dwg.add(svgwrite.shapes.Line(
            (0+margin_ltbr[0], 3*font_height),
            (critical_distance+margin_ltbr[0], 3*font_height),
            stroke=critical_line_color)
    )

    dwg.add(svgwrite.shapes.Line(
            (0+margin_ltbr[0], 2.75*font_height),
            (0+margin_ltbr[0], 3.25*font_height),
            stroke=critical_line_color)
    )

    dwg.add(svgwrite.shapes.Line(
            (critical_distance+margin_ltbr[0], 2.75*font_height),
            (critical_distance+margin_ltbr[0], 3.25*font_height),
            stroke=critical_line_color)
    )

    dwg.add(svgwrite.shapes.Line(
            (0+margin_ltbr[0], 4+font_height+interval_height+margin_ltbr[1]),
            (size+margin_ltbr[0], 4+font_height+interval_height+margin_ltbr[1]),
            stroke=main_line_color)
    )

    dwg.save()


def visualize_posthoc_nemenyi_test2(methods_with_score, num_exp,  alpha=0.05, filename='test.svg'):
    '''
    20220330,
    ref: 1. https://orange3.readthedocs.io/projects/orange-data-mining-library/en/latest/reference/evaluation.cd.html#
    :param methods_with_score:
    :param num_exp:
    :param alpha:
    :return:
    '''
    names = list(methods_with_score.keys())
    avranks = list(methods_with_score.values())

    # cmpt Critical Difference (CD)
    cd = Orange.evaluation.compute_CD(avranks, num_exp, alpha=str(alpha))
    print('Critical difference (CD): {}'.format(cd))

    graph_ranks(avranks, names, cd=cd, width=6, textspace=1.5, filename=filename)


if __name__ == '__main__':
    methods_and_scores = {
        'BR': 3,
        'LP': 4,
        'mlg-lp-unweighted': 2.4444,
        'mlg-lp': 1.7778,
        'RAKEL1': 7,
        'RAKEL2': 5.8889,
        'MLkNN': 6.1111,
        'BPMLL': 6.7778,
        'CLR': 8,
    }
    #visualize_posthoc_nemenyi_test(methods_and_scores)
    visualize_posthoc_nemenyi_test2(methods_and_scores, num_exp=5, alpha=0.05, filename='test2.svg')



