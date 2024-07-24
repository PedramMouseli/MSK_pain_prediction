#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 21:30:04 2022

@author: moayedilab
"""
import numpy as np
from sklearn.linear_model import LinearRegression
import antropy as ant

import time
import matplotlib.pyplot as plt
import pywt
import seaborn as sns
import matplotlib.gridspec as gridspec
import pandas as pd
from mlxtend.evaluate import permutation_test

def normalize(data, base_events=[], baseline=False, mean=True, std=True, sep_baseline=False):
    """
    Normalize the signal.

    Parameters
    ----------
    data : array
        Signal to normalize.
        
    mean : Bool, optional
        mean subtraction. The default is True.
        
    std : Bool, optional
        dividing by standard deviation. The default is True.

    Returns
    -------
    data : array
        Normalized signal.

    """
    if sep_baseline:
        data[0] -= data[1].mean()
        data[0] /= data[1].std()
        data = data[0]
    else:      
        if mean:
            if baseline:
                data -= data[base_events[0]:base_events[1]].mean()
            else:
                data -= data.mean()
        if std:
            data /= data.std()
    
    return data

def load_ratings(participants_df, ratings_path):
    ratings = pd.read_csv(ratings_path)

    ratings = ratings.rename(columns={"Please enter participant ID and click submit.":"sub_id"})

    pain_df = pd.DataFrame()
    fatigue_df = pd.DataFrame()

    for subject_id in participants_df['sub_id']:

        sub_num = participants_df.loc[participants_df['sub_id']==subject_id]['sub_num']
        
        pain_rc = np.zeros(16)
        pain_lc = np.zeros(16)
        pain_rt = np.zeros(16)
        pain_lt = np.zeros(16)
        
        fat_rc = np.zeros(16)
        fat_lc = np.zeros(16)
        fat_rt = np.zeros(16)
        fat_lt = np.zeros(16)
        
        pain_rc[0] = int(ratings.loc[ratings['sub_id']==subject_id]['Right Cheek'])
        pain_lc[0] = int(ratings.loc[ratings['sub_id']==subject_id]["Left Cheek"])
        pain_rt[0] = int(ratings.loc[ratings['sub_id']==subject_id]["Right Temple"])
        pain_lt[0] = int(ratings.loc[ratings['sub_id']==subject_id]["Left Temple"])
        
        for i in range(1,32):
            if (i % 2) == 0:
                pain_rc[int(i/2)] = int(ratings.loc[ratings['sub_id']==subject_id]["Right Cheek."+str(i)])
                pain_lc[int(i/2)] = int(ratings.loc[ratings['sub_id']==subject_id]["Left Cheek."+str(i)])
                pain_rt[int(i/2)] = int(ratings.loc[ratings['sub_id']==subject_id]["Right Temple."+str(i)])
                pain_lt[int(i/2)] = int(ratings.loc[ratings['sub_id']==subject_id]["Left Temple."+str(i)])
            else:
                fat_rc[int((i-1)/2)] = int(ratings.loc[ratings['sub_id']==subject_id]["Right Cheek."+str(i)])
                fat_lc[int((i-1)/2)] = int(ratings.loc[ratings['sub_id']==subject_id]["Left Cheek."+str(i)])
                fat_rt[int((i-1)/2)] = int(ratings.loc[ratings['sub_id']==subject_id]["Right Temple."+str(i)])
                fat_lt[int((i-1)/2)] = int(ratings.loc[ratings['sub_id']==subject_id]["Left Temple."+str(i)])
        
        sub_pain = pd.DataFrame({'subject':np.repeat(sub_num,len(pain_rc)*4), 'trial':np.tile(range(1,17),4),
                                 'muscle':np.repeat(['right masseter','left masseter','right temporalis','left temporalis'],16),
                                 'pain':np.r_[pain_rc, pain_lc, pain_rt, pain_lt]})
        pain_df = pd.concat([pain_df, sub_pain], ignore_index=True)
        
        sub_fatigue = pd.DataFrame({'subject':np.repeat(sub_num,len(pain_rc)*4), 'trial':np.tile(range(1,17),4),
                                 'muscle':np.repeat(['right masseter','left masseter','right temporalis','left temporalis'],16),
                                 'fatigue':np.r_[fat_rc, fat_lc, fat_rt, fat_lt]})
        fatigue_df = pd.concat([fatigue_df, sub_fatigue], ignore_index=True)
        
    return pain_df, fatigue_df

def nirs_features(sub_list, nirs_clench, measure='tsi', side='r', sensor='1'):
    
    task_median = np.zeros([len(sub_list),15])
    rest_median = np.zeros([len(sub_list),15])
    pre_rest_median = np.zeros(len(sub_list))

    for j,sub in enumerate(sub_list):
        pre_rest_median[j] = np.median(np.array(nirs_clench.loc[sub][side+"_"+measure+sensor+"_pre_rest2"]))
        
        for i in range(1,16):
            task_median[j,i-1] = np.median(np.array(nirs_clench.loc[sub][side+"_"+measure+sensor+"_task_"+str(i)])[250:1250])
            
            rest_median[j,i-1] = np.median(np.array(nirs_clench.loc[sub][side+"_"+measure+sensor+"_rest_"+str(i)])[250:1250])
            
         
    median_diff = task_median - rest_median

         
    ##### fit a linear model to differences
    diff_coef = np.zeros(len(sub_list))

    for j,sub in enumerate(sub_list):
        y = median_diff[j,:16]
        x = np.array(range(len(y))).reshape(-1,1)
        reg = LinearRegression().fit(x, y)
        diff_coef[j] = reg.coef_[0]
        
    return diff_coef, task_median, pre_rest_median


def entropy(sub_list, emg_clench, side='r'):
    
    task_ent = np.zeros([len(sub_list),15])

    for j,sub in enumerate(sub_list[-1:]):
        start = time.time()
        # for a few subjects data collected at 2000 Hz sampling frequency, for others it's 2048 Hz
        sf = int(len(emg_clench.loc[sub][side+"_task_1"])/30)
        
        for i in range(1,16):
            y = np.array(emg_clench.loc[sub][side+"_task_"+str(i)])[5*sf:25*sf]
            task_ent[j,i-1] = ant.sample_entropy(y, order=2)
            
        end = time.time()
        print(sub + ' took ' + str(round(end-start, 3)) + ' seconds\n')
            
    ##### fit a linear model to entropy values
    ent_coef_task = np.zeros(len(sub_list))

    for j,sub in enumerate(sub_list):
        y_task = task_ent[j,:]
        x = np.array(range(len(y_task))).reshape(-1,1)/100
        reg_task = LinearRegression().fit(x, y_task)
        ent_coef_task[j] = reg_task.coef_[0]
            
    return task_ent, ent_coef_task


def extraxt_wt_power(sub_list, emg_clench, side='r'):
    
    task_wt = np.zeros([len(sub_list),9,15])

    for j,sub in enumerate(sub_list):
        # for a few subjects data collected at 2000 Hz sampling frequency, for others it's 2048 Hz
        sf = int(len(emg_clench.loc[sub][side+"_task_1"])/30)
        
        for i in range(1,16):
            task_wt[j,:,i-1] = wt_power(np.array(emg_clench.loc[sub][side+"_task_"+str(i)])[5*sf:25*sf])
            
    ##### fit a linear model to wt power values
    wt_coef_task = np.zeros([len(sub_list),9])


    for j,sub in enumerate(sub_list):
        for i in range(9):
            y_task = task_wt[j,i,:]
            x = np.array(range(len(y_task))).reshape(-1,1)/100
            reg_task = LinearRegression().fit(x, y_task)
            wt_coef_task[j,i] = reg_task.coef_[0]
            
    return wt_coef_task



def wt_power(x, wavelet='db5'):
    x = np.asarray(x)
    list_coeff = pywt.wavedec(x, wavelet)
    power = []
    for i in range(4,13):
        power.append(np.sum(list_coeff[i]**2))
        
    power = np.array(power)
    
    return power


def plot_wavelet(time, signal, scales, 
                 waveletname = 'cgau5', 
                 cmap = plt.cm.seismic, 
                 title = 'Wavelet Transform (Power Spectrum) of signal', 
                 ylabel = 'Frequency', 
                 xlabel = 'Time'):
    
    dt = time[1] - time[0]
    [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)
    power = (abs(coefficients)) ** 2
    period = frequencies
    levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8]
    contourlevels = np.log2(levels)
    
    fig, ax = plt.subplots(figsize=(15, 10))
    im = ax.contourf(time, np.log2(period), np.log2(power), contourlevels, extend='both',cmap=cmap)
    
    ax.set_title(title, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=18)
    
    yticks = 2**np.arange(np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max())))
    ax.set_yticks(np.log2(yticks))
    ax.set_yticklabels(yticks)
    # ax.invert_yaxis()
    # ylim = ax.get_ylim()
    # ax.set_ylim(ylim[0], -1)
    # plt.show()
    
    cbar_ax = fig.add_axes([0.95, 0.5, 0.03, 0.25])
    fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    plt.show()
    

class SeabornFig2Grid():

    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
            isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
            self._finalize()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
            self._finalize()
        else:
            # print(gridspec.GridSpecFromSubplotSpec(1,1, subplot_spec=self.subplot)[0].shape)
            self.sg[0].set_size_inches(self.fig.get_size_inches())
            self._moveaxes(self.sg[1], gridspec.GridSpecFromSubplotSpec(1,1, subplot_spec=self.subplot)[0])
            plt.close(self.sg[0])
            self.fig.canvas.mpl_connect("resize_event", self._resize)
            self.fig.canvas.draw()
        # self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())
        
        
def plot_predictions(coefs, target_df, modality, target, plot_path, save_plot=False):
    feature_labels = ['sign(NIRS slope)', 'abs(NIRS slope)', 'normalized median 1-3', 'normalized median 4-6', 'normalized median 7-9', 'normalized median 10-12', 'normalized median 13-15',
                      'sign(entropy slope)', 'abs(entropy change)', 'sign(DWT slope) level 9', 'sign(DWT slope) level 8', 'sign(DWT slope) level 7', 'sign(DWT slope) level 6', 'sign(DWT slope) level 5',
                      'sign(DWT slope) level 4', 'sign(DWT slope) level 3', 'sign(DWT slope) level 2', 'sign(DWT slope) level 1']
    print('modality: ', modality)
    if modality == 'EMG':
        labels = feature_labels[7:]
    if modality == 'NIRS':
        labels = feature_labels[:7]
    if modality == 'NIRS_EMG':
        labels = np.array(feature_labels)

    ## predictions
    corr = np.corrcoef(target_df['pred_targets'],target_df['targets'])[0,1]
    corr_male = np.corrcoef(target_df.loc[target_df['sex']=='Male', 'pred_targets'],target_df.loc[target_df['sex']=='Male', 'targets'])[0,1]
    corr_female = np.corrcoef(target_df.loc[target_df['sex']=='Female', 'pred_targets'],target_df.loc[target_df['sex']=='Female', 'targets'])[0,1]

    p_val = permutation_test(target_df['pred_targets'],target_df['targets'],
                               func=lambda x, y: np.abs(np.corrcoef(x, y))[0,1],
                               method='approximate',
                               num_rounds=10000,
                               seed=42)
        
    g0 = sns.jointplot(data=target_df, x="targets", y="pred_targets", hue="sex", space=0.5, color ='tab:blue', 
                       xlim=(target_df['targets'].min()-5, target_df['targets'].max()+5),
                       ylim=(target_df['pred_targets'].min()-5, target_df['pred_targets'].max()+5), joint_kws={'s':100}, ratio=3)

    g1 = sns.regplot(x='targets', y='pred_targets',data=target_df, color ='tab:blue', scatter_kws={'alpha':0.5, 's':100, "zorder":-1})

    # pred_range = target_df['pred_targets'].max() - target_df['pred_targets'].min()
    g1.text(target_df['targets'].max()-25, target_df['pred_targets'].min(), 'r = '+str(round(corr,2))+ ',  p = '+str(round(p_val,4))+'\n'+r'$r_{male}$ = '+str(round(corr_male,2))+r', $r_{female}$ = '+str(round(corr_female,2)),
            fontsize = 13,          # Size
            fontstyle = "oblique",  # Style
            color = "black",          # Color
            ha = "center") # Horizontal alignment
            # va = "center") # Vertical alignment
    plt.xlabel(f'{target}',fontsize=13)
    if target == 'Pain change':
        plt.ylabel(f'Predicted {target.lower()}',fontsize=13)
    else:
        plt.ylabel(f'Predicted {target}',fontsize=13)
    # remove hue legend title
    g1.legend_.set_title(None)

    ## weights
    weights_dict = {'fold':np.array(range(1,11))}
    for i,feature in enumerate(labels):
        weights_dict[feature] = coefs[:,i]
    weights_df = pd.DataFrame(weights_dict)
    weights_df = pd.melt(weights_df, id_vars=['fold'], var_name='features', value_name='weights')

    fig1, axes = plt.subplots(1,1)
    sns.set_style("white")
    w_plot = sns.barplot(data=weights_df, x="weights", y="features", errorbar="ci", capsize=.1, edgecolor=".3", palette = "colorblind", orient="h", ax=axes)
    _, xlabels = plt.xticks()
    # w_plot.set_xticklabels(xlabels, size = 13)
    _, ylabels = plt.yticks()
    w_plot.set_yticklabels(ylabels, size = 13)
    # plt.xticks(rotation=70)
    plt.xlabel('Average weight', fontsize = 13)
    plt.ylabel("", fontsize = 12)
    plt.axvline(x=0, color='black', linestyle='-')
    sns.despine(bottom = True, left = False)


    fig = plt.figure(figsize=(9,6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[0.25, 0.75])
    # gs = gridspec.GridSpec(2, 1, height_ratios=[0.6, 0.4])

    mg0 = SeabornFig2Grid(g0, fig, gs[1])
    plt.close(g0.fig)
    # mg0 = SeabornFig2Grid(g1, fig, gs[1])
    mg1 = SeabornFig2Grid([fig1,axes], fig, gs[0])

    gs.tight_layout(fig)

    plt.show()
    
    if save_plot:
        fig.savefig(f'{plot_path}/prediction_{target}_{modality}.png',dpi=fig.dpi, bbox_inches = "tight")
        
def plot_feature_selection(support, modality, plot_path, save_plot=False):
    feature_labels = ['sign(NIRS slope)', 'abs(NIRS slope)', 'normalized median 1-3', 'normalized median 4-6', 'normalized median 7-9', 'normalized median 10-12', 'normalized median 13-15',
                      'sign(entropy slope)', 'abs(entropy change)', 'sign(DWT slope) level 9', 'sign(DWT slope) level 8', 'sign(DWT slope) level 7', 'sign(DWT slope) level 6', 'sign(DWT slope) level 5',
                      'sign(DWT slope) level 4', 'sign(DWT slope) level 3', 'sign(DWT slope) level 2', 'sign(DWT slope) level 1']
    if modality == 'EMG':
        labels = feature_labels[7:]
    if modality == 'NIRS':
        labels = feature_labels[:7]
    if modality == 'NIRS_EMG':
        labels = np.array(feature_labels)
        
    occurences = np.sum(support,axis=0)
    feat_occu_df = pd.DataFrame({'label': labels, 'occurence': occurences})
    feat_occu_sorted = feat_occu_df.sort_values(['occurence'], ascending=False).reset_index(drop=True)

    plt.figure('feature importance',figsize=[10,5])

    ax = sns.barplot(x='label', y='occurence', data=feat_occu_sorted, palette="colorblind", edgecolor=".3", errorbar=None)
    ax.bar_label(ax.containers[0], fontsize=10)

    plt.xticks(rotation=70)
    # plt.tight_layout()
    # plt.axhline(y=0.5*len(support), color='r', linestyle='--')
    plt.xlabel('', fontsize = 2)
    plt.ylabel('Occurence', fontsize = 13)
    sns.despine(right = True, top = True)
    plt.tick_params(bottom = True, left = True)
    
    plt.show()
    
    if save_plot:
        plt.savefig(f'{plot_path}/selected_features_{modality}.png',dpi=600)
        