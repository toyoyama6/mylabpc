import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math


class Pulse():
    
    def cal_time(self, ts, ys, upper_threshold, lower_threshold, ymax=None, ymin=None, tr_start_min=None, ton_min=None):
        dst = {}

        if tr_start_min == None:
            tr_start_min = min(ts)

        if ton_min == None:
            ton_start = 0

        if ymax ==None:
            ymax = max(ys)

        if ymin == None:
            ymin = min(ys)

        dst['ymax'] = ymax
        dst['ymin'] = ymin
        dst['yamp'] = yamp = ymax - ymin
        dst['upper'] = upper_threshold
        dst['lower'] = lower_threshold

        trs_start = ts[ys > yamp * lower_threshold]
        dst['tr_start'] = tr_start = trs_start[trs_start > tr_start_min].min()

        trs_end = ts[ys > yamp * upper_threshold]
        dst['tr_end'] = tr_end = trs_end[trs_end > tr_start].min()

        dst['tr'] = tr_end - tr_start

        tws_start = ts[ys > yamp * 0.5]
        dst['tw_start'] = tw_start = tws_start[tws_start > tr_start_min].min()

        tws_end = ts[ys < yamp * 0.5]
        dst['tw_end'] = tw_end = tws_end[tws_end > tw_start].min()

        dst['tw'] = tw_end - tw_start

        # try:
        #     tfs_start = ts[ys < yamp * upper_threshold]
        #     dst['tf_start'] = tf_start =tfs_start[tfs_start > tr_end + ton_min].min()
        # except:
        #     dst['tf_start'] = tf_start = ts[-1]

        # dst['ton'] = ton = tf_start - tr_end

        # try:
        #     tfs_end = ts[ys < yamp * lower_threshold]
        #     dst['tf_end'] = tf_end = tfs_end[tfs_end > tf_start].min()
        # except:
        #     dst['tf_end'] = tf_end = ts[-1]

        # dst['tf'] = tf = tf_end - tf_start

        # taus = ts[ys >= yamp * 0.632]
        # dst['tau'] = tau = taus[taus > tr_start_min].min() - tr_start
        self.pulse_times = dst

        return dst
    

    def show_time(self):
        dst = {}
        tr_start = self.pulse_times["tr_start"]
        tr_end = self.pulse_times["tr_end"]
        tr = self.pulse_times["tr"]
        tw_start = self.pulse_times["tw_start"]
        tw_end = self.pulse_times["tw_end"]
        tw = self.pulse_times["tw"]
        # ton = self.pulse_times["ton"]
        # tf_start = self.pulse_times["tf_start"]
        # tf_end = self.pulse_times["tf_end"]
        # tf = self.pulse_times["tf"]
        # tau = self.pulse_times["tau"]
        # ymax = self.pulse_times["ymax"]
        # ymin = self.pulse_times["ymin"]
        # yamp = self.pulse_times["yamp"]

        print("Tr start:", tr_start)
        print("Tr end:", tr_end)
        print("Tr:", tr)
        print('Tw start:', tw_start)
        print('Tw end:', tw_end)
        print('Tw:', tw)
        # print("Ton:", ton)
        # print("Tf start:", tf_start)
        # print("Tf end:", tf_end)
        # print("Tf:", tf)
        # print("Tau:", tau)


    def save_graph(self, x, y, xlabel, ylabel, save_dir, file_name, label_name='Y'):

        tr_start = self.pulse_times["tr_start"]
        tr_end = self.pulse_times["tr_end"]
        tr = self.pulse_times["tr"]
        tw_start = self.pulse_times["tw_start"]
        tw_end = self.pulse_times["tw_end"]
        tw = self.pulse_times["tw"]
        # ton = self.pulse_times["ton"]
        # tf_start = self.pulse_times["tf_start"]
        # tf_end = self.pulse_times["tf_end"]
        # tf = self.pulse_times["tf"]
        # tau = self.pulse_times["tau"]
        ymax = self.pulse_times["ymax"]
        ymin = self.pulse_times["ymin"]
        yamp = self.pulse_times["yamp"]
        upper_threshold = self.pulse_times['upper']
        lower_threshold = self.pulse_times['lower']

        plt.plot(x, y, color='r', alpha=0.7, ms=2, label=label_name)

        plt.vlines(tr_start, min(y), max(y), ls='--', color='blue', lw=1, label='Tr start')
        plt.vlines(tr_end, min(y), max(y), ls='--', color='g', lw=1, label='Tr end')
        # plt.vlines(tf_start, min(y), max(y), ls='--', lw=1, label='Tf start')
        # plt.vlines(tf_end, min(y), max(y), ls='--', color='m', lw=1, label='Tf end')
        # plt.vlines(tau, min(y), max(y), ls='-', lw=1, label='Tau')
        plt.vlines(tw_start, min(y), max(y), ls='--', color='black', lw=1, label='Tw start')
        plt.vlines(tw_end, min(y), max(y), ls='--', color='black', lw=1, label='Tw end')

        plt.hlines(yamp * upper_threshold, min(x), max(x), ls='--', color='r', lw=1, label='Amp '+str(upper_threshold*100)+'%')
        plt.hlines(yamp * lower_threshold, min(x), max(x), ls='--', color='y', lw=1, label='Amp '+str(lower_threshold*100)+'%')
        plt.hlines(yamp * 0.5, min(x), max(x), ls='--', color='g', lw=1, label='Amp 50%')

        plt.legend(loc='best')
        plt.xlabel(xlabel=xlabel, fontsize=12)
        plt.ylabel(ylabel=ylabel, fontsize=12)
        plt.savefig(save_dir+file_name)
        #plt.show()


def func(x, *params):

    #paramsの長さでフィッティングする関数の数を判別。
    num_func = int(len(params)/3)

    #ガウス関数にそれぞれのパラメータを挿入してy_listに追加。
    y_list = []
    for i in range(num_func):
        y = np.zeros_like(x)
        param_range = list(range(3*i,3*(i+1),1))
        amp = params[int(param_range[0])]
        ctr = params[int(param_range[1])]
        wid = params[int(param_range[2])]
        y = y + amp * np.exp( -((x - ctr)/wid)**2)
        y_list.append(y)

    #y_listに入っているすべてのガウス関数を重ね合わせる。
    y_sum = np.zeros_like(x)
    for i in y_list:
        y_sum = y_sum + i

    #最後にバックグラウンドを追加。
    y_sum = y_sum + params[-1]

    return y_sum

def fit_plot(x, *params):
    num_func = int(len(params)/3)
    y_list = []
    for i in range(num_func):
        y = np.zeros_like(x)
        param_range = list(range(3*i,3*(i+1),1))
        amp = params[int(param_range[0])]
        ctr = params[int(param_range[1])]
        wid = params[int(param_range[2])]
        y = y + amp * np.exp( -((x - ctr)/wid)**2) + params[-1]
        y_list.append(y)
    return y_list




    