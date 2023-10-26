import matplotlib as mpl
import collections
from matplotlib.ticker import ScalarFormatter, NullFormatter, PercentFormatter
from matplotlib import gridspec
import matplotlib.cm as cm
import sys, os
import numpy as np
import pylab
import matplotlib.pyplot as plt
import pickle, pickle
import scipy.stats
from scipy.optimize import leastsq
import scipy
import statsmodels.api as sm
import pandas as pd
from statsmodels.formula.api import ols
from copy import deepcopy
import matplotlib.pyplot as plt
import metrics_mod

params = { 'figure.figsize': (8, 10),
           'legend.fontsize': 8,
           # 'title.fontsize': 8,
           'lines.color':'black',
           'lines.linewidth':1,
            'xtick.labelsize':4,
            'xtick.major.pad'  : 3, 
            'xtick.major.size' : 2,
            'ytick.major.pad'  : 3,
            'ytick.major.size' : 2,
            'ytick.labelsize':7,
            'axes.labelsize':8,
            'font.size':8,
            'axes.labelpad':3,
            # 'text.usetex' : True,
            # 'legend.labelsep': 0.0005 
            'pdf.fonttype' : 42,
            'figure.dpi': 600,
            }
mpl.rcParams.update(params)
def ylabl( text, axtrans ):
   pylab.text(-0.15, 0.5,text, fontsize = 10, horizontalalignment='center',
   verticalalignment='center',rotation='vertical',
   transform = axtrans)
def ylablsup( text, axtrans ):
   pylab.text(-0.10, 0.5,text, fontsize = 8, horizontalalignment='center',
   verticalalignment='center',rotation='vertical',
   transform = axtrans)
def ylabr( text, axtrans ):
   pylab.text(1.15, 0.5,text,fontsize = 10, horizontalalignment='center',
   verticalalignment='center',rotation='vertical',
   transform = axtrans)

def AU_ROC(data1, data2):
    """ AUROC along first axis for two samples of equal length as computed from MannWhitneyU/N^2, 
        see http://en.wikipedia.org/wiki/Mann-Whitney_U"""
    samples = float(len(data1))
    if np.rank(data1) == 1:
        data1 = data1.reshape(samples,1)
        data2 = data2.reshape(samples,1)
    auroc = 1. - (np.vstack((data1,data2)).argsort(axis=0).argsort(axis=0)[:samples].sum(axis=0) - samples*(samples+1.)/2.)/samples**2
    auroc[np.any(np.isnan(data1), axis=0)*np.any(np.isnan(data2), axis=0)] = np.nan
    return auroc

save_type = 'pdf'
folder_name = './' #SAVE PATH OF NUMERICAL EXP
save_folder = './' #FOLDER OF OUTPUT FIGURES

test = False

try:
    arg = sys.argv
    ci_test = str(arg[1])  
    variant = str(arg[2])
except:
    arg = ''
    ci_test = 'par_corr'
    variant = 'boot_rea_highdegree'

print(variant)
name = {'par_corr':r'ParCorr', 'gp_dc':r'GPDC', 'cmi_knn':r'CMIknn'}

def get_metrics_from_file(para_setup):

    name_string = '%s-'*len(para_setup)  # % para_setup
    name_string = name_string[:-1]

    try:
        print("load from metrics file  %s_link_frequency_metrics.dat " % (folder_name + name_string % tuple(para_setup)))
        results = pickle.load(open(folder_name + name_string % tuple(para_setup) + '_link_frequency_metrics.dat', 'rb'), encoding='latin1')
    except Exception as e:
        print('failed from metrics file '  , tuple(para_setup))
        print(e)
        return None

    return results

def draw_it(paras, which, suffix=""):

    figsize = (3, 2.5)  #(4, 2.5)
    capsize = .5
    marker1 = 'o'
    marker2 = 's'
    marker3 = '+'
    alpha_marker = 1.

    params = { 
           'legend.fontsize': 5,
           'legend.handletextpad': .05,
           # 'title.fontsize': 8,
           'lines.color':'black',
           'lines.linewidth':.5,
           'lines.markersize':2,
           # 'lines.capsize':4,
            'xtick.labelsize':4,
            'xtick.major.pad'  : 1, 
            'xtick.major.size' : 2,
            'ytick.major.pad'  : 1,
            'ytick.major.size' : 2,
            'ytick.labelsize':4,
            'axes.labelsize':8,
            'font.size':8,
            'axes.labelpad':2,
            # 'axes.grid': True,
            'axes.spines.right' : False,
            'axes.spines.top' : False,

            }
    mpl.rcParams.update(params)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2)
    ax1a = fig.add_subplot(gs[0, 0])
    ax1b = fig.add_subplot(gs[1, 0])

    ax2a = fig.add_subplot(gs[0, 1])
    ax2b = fig.add_subplot(gs[1, 1])


    for method in methods:
        for idx,para in enumerate(paras):
            para_plot = para
            N_here = N
            auto_here = auto
            tau_max_here = tau_max
            frac_unobserved_here = frac_unobserved
            pc_alpha_here = pc_alpha
            T_here = T

            n_links_here = links_from_N(N_here)

            para_setup = (model, N_here, n_links_here, min_coeff, coeff, auto_here, contemp_fraction, frac_unobserved_here,  
                            max_true_lag, T_here, ci_test, method, pc_alpha_here, tau_max_here, para, N_draw_bs) 
            metrics_dict = get_metrics_from_file(para_setup)
            if metrics_dict is not None:
                color_ = color_picker(method,idx)
                ax1a.errorbar(para_plot, metrics_dict['adj_lagged_abs_freq_diff'+suffix][0],
                    yerr= metrics_dict['adj_lagged_abs_freq_diff'+suffix][1],
                    capsize=capsize, alpha=alpha_marker,
                    color=color_, marker=marker2, linestyle='solid',elinewidth=1)
                print(metrics_dict['adj_lagged_abs_freq_diff'+suffix][0])
                ax1b.errorbar(para_plot, metrics_dict['adj_contemp_abs_freq_diff'+suffix][0], 
                            yerr = metrics_dict['adj_contemp_abs_freq_diff'+suffix][1],
                            capsize=capsize,  alpha=alpha_marker,
                            color=color_, marker=marker2, linestyle='solid',elinewidth=1)

                ax2a.errorbar(para_plot, metrics_dict['adj_anylink_abs_freq_diff'+suffix][0],
                            yerr = metrics_dict['adj_anylink_abs_freq_diff'+suffix][1],
                            capsize=capsize, alpha=alpha_marker,
                            color=color_, marker=marker2, linestyle='solid',elinewidth=1)

                ax2b.errorbar(para_plot, metrics_dict['computation_time'][0], 0.5*(metrics_dict['computation_time'][1][1]-metrics_dict['computation_time'][1][0]), capsize=capsize, alpha=alpha_marker,
                    color=color_, marker='p', linestyle='solid',elinewidth=1)
                
    axes = {'ax1a':ax1a, 'ax1b':ax1b, 'ax2a':ax2a, 'ax2b':ax2b}
    for axname in axes:

        ax = axes[axname]
        ax.set_xlim(0, paras[-1])
        ax.xaxis.set_ticks([p for p in paras])
        ax.xaxis.set_ticklabels([str(p) for p in paras])

        for line in ax.get_lines():
            line.set_clip_on(False)

        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.spines['left'].set_position(('outward', 3))
        ax.spines['bottom'].set_position(('outward', 3))
        ax.grid(axis='y', linewidth=0.3)

        pad = 2
        xmin= min(paras)-5
        xmax=max(paras)+25
        if axname == 'ax1a':
            ax.set_title('Lagged Abs. Freq. Error', fontsize=6, pad=pad)
            #ax.set_ylim(0., 1.)
            ax.set_xlim(xmin, xmax)
            ax.tick_params(labelbottom=False)    
            vals = ax.get_yticks()
            ax.set_yticklabels(['{:,.1%}'.format(x/100) for x in vals])

        elif axname == 'ax1b':
            ax.set_title('Contemp. Abs. Freq. Error', fontsize=6, pad=pad)
            #ax.set_ylim(0., 1.)
            ax.set_xlim(xmin, xmax)
            vals = ax.get_yticks()
            ax.set_yticklabels(['{:,.1%}'.format(x/100) for x in vals])
            xlabels = ax.get_xticklabels()
            ax.set_xticklabels(xlabels, rotation=75, horizontalalignment='center')

        elif axname == 'ax2a':
            ax.set_title('All links Abs. Freq. Error', fontsize=6, pad=pad)
            #ax.set_ylim(0., 1.)
            ax.set_xlim(xmin, xmax)
            vals = ax.get_yticks()
            ax.set_yticklabels(['{:,.1%}'.format(x/100) for x in vals])
            ax.tick_params(labelbottom=False)  

        elif axname == 'ax2b':
            ax.set_title('Runtime [s]', fontsize=6, pad=pad)
            # ax.set_ylim(0., 1.)
            ax.set_xlim(xmin, xmax)
            #ax.tick_params(labelbottom=False)    
            #ax.set_yscale('log')
            xlabels = ax.get_xticklabels()
            ax.set_xticklabels(xlabels, rotation=75, horizontalalignment='center')

    axlegend = fig.add_axes([0.05, .89, 1., .05])
    axlegend.axis('off')

    if 'random_lineargaussian' in model:
        model_name = r"$\mathcal{N}$"
    elif 'random_linearmixed' in model:
        model_name = r"$\mathcal{N{-}W}$"
    elif 'random_nonlinearmixed' in model:
        model_name = r"$\mathcal{N{-}W}^2$"
    elif 'random_nonlineargaussian' in model:
        model_name = r"$\mathcal{N}^2$"
    else:
        model_name = model
    if 'fixeddensity' in model:
        model_name += r"$_{d{=}0.3}$"
    elif 'highdegree' in model:
        model_name += r"$_{d{=}1.5}$"

    if which == 'n_bs':
        plt.figtext(0.5, 0.0, r"Number of bootstrap realizations", fontsize=6,
            horizontalalignment='center', va='bottom')
        plt.figtext(1., 1., r"%s: N=%d, $T=%d, a=%s$" %(model_name, N, T, auto) 
                            +"\n" + r"%s, $\alpha=%s, \tau_{\max}=%d$" %(name[ci_test],pc_alpha,tau_max),
         fontsize=6, ha='right', va='top')

    if suffix=="_existing":
        plt.title("Mean Absolute Frequency Error (Exising links)",loc="left",fontsize=7,y=0.5)
    elif suffix=="_absent":
        plt.title("Mean Absolute Frequency Error (Absent links)",loc="left",fontsize=7,y=0.5)
    else:
        plt.title("Mean Absolute Frequency Error",loc="left",fontsize=7,y=0.5)
    plt.figtext(0., 1., "B", fontsize=12, fontweight='bold',ha='left', va='top')
    fig.subplots_adjust(left=0.08, right=0.97, hspace=.3, bottom=0.12, top=0.85, wspace=.3)
    print("Saving plot to %s" %(save_folder + '%s_mean_abs_freq_error%s.%s' %(save_suffix, suffix, save_type)))
    fig.savefig(save_folder + '%s_mean_abs_freq_error%s.%s' %(save_suffix, suffix, save_type))
    plot_files.append(save_folder + '%s_mean_abs_freq_error%s.%s' %(save_suffix, suffix, save_type))

def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

def color_picker(method,idx=0):
    if "bootstrap" in method:
        color= adjust_lightness('green', amount=2.5)
        #color_scaling = [adjust_lightness('green', amount=i) for i in [3.5,2.9,2.3,1.7,1.1,0.5,0]]
        return color
    else: return 'orange'


def method_label(method):
    # return method
    if not 'paper' in variant:
        return method

    if 'standard_pcmci' in method:
        if 'allpx0' in method:
            return r'PCMCI$^+_0$'
        elif 'laggedpx0' in method:
            return r'PCMCI$^+_{0-}$'
        elif 'resetlagged' in method:
           return r'PCMCI$^+_{\rm reset}$'
        else:
            return r'PCMCI$^+$'

    elif 'residualPC' in method or 'GCresPC' in method:
        return r"GCresPC"
    elif 'lingam' in method:
         return r"LiNGAM" 
    elif 'pcalg' in method:
         return r"PC" 
    else:
        return method

def links_from_N(num_nodes):
    if num_nodes == 2:
        return 1

    if 'fixeddensity' in model:
        return max(num_nodes, int(0.2*num_nodes*(num_nodes-1.)/2.)) 
    elif 'highdegree' in model:
        return int(1.5*num_nodes) 
    else:
        return num_nodes



if __name__ == '__main__':


    save_type = 'pdf'
    plot_files = []
    fpr_precision = 'fpr'

    if 'versions' in variant:
        methods = [

            'bootstrap_pcmci+'
            ]
    else:
        methods = []

        methods += [
            'bootstrap_pcmci+',

            ]


    if ci_test == 'par_corr':
        if 'mixed' in variant:
            model  = 'random_linearmixed'  # random_lineargaussian random_linearmixed random_nonlinearmixed
        else:
            model  = 'random_lineargaussian'  # random_lineargaussian random_linearmixed random_nonlinearmixed
    else:
        if 'mixed' in variant:
            model  = 'random_nonlinearmixed'  # random_lineargaussian random_linearmixed random_nonlinearmixed
        else:
            model  = 'random_nonlineargaussian'  # random_lineargaussian random_linearmixed random_nonlinearmixed

    if 'fixeddensity' in variant:
        model += '_fixeddensity'
    elif 'highdegree' in variant:
        model += '_highdegree'

    if 'boot_rea' in variant:
        T_here= [500]
        N_here= [3]
        num_rows = 3
        tau_max = 2
        auto_here= [0.95]  #[0.95]      
        min_coeff = 0.1
        coeff = 0.5
        N_draw_bs= 1
        frac_unobserved = 0.
        contemp_fraction = 0.3
        max_true_lag = 2
        vary_n_bs= [25,100,250,500,750,1000,1500,2000,2500]
        pc_alpha_here = [0.01]
        for auto in auto_here:
            for T in T_here: 
                for N in N_here:
                    n_links = links_from_N(N)  
                    for pc_alpha in pc_alpha_here:
                        for suffix in ["","_existing","_absent"]:
                            para_setup_name = (variant, N, n_links, min_coeff, coeff, contemp_fraction, frac_unobserved,  
                                                max_true_lag, T, ci_test, pc_alpha, tau_max, N_draw_bs) 
                            save_suffix = '%s-'*len(para_setup_name) % para_setup_name
                            save_suffix = save_suffix[:-1]
                            print(save_suffix)
                            draw_it(paras=vary_n_bs, which='n_bs', suffix=suffix)
