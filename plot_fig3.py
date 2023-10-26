import matplotlib as mpl
import collections
from matplotlib.ticker import ScalarFormatter, NullFormatter, PercentFormatter
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter
import matplotlib.cm as cm

params = { 'figure.figsize': (8, 10),
           'legend.fontsize': 8,
           'lines.color':'black',
           'lines.linewidth':1,
            'xtick.labelsize':10,
            'xtick.major.pad'  : 3, 
            'xtick.major.size' : 2,
            'ytick.major.pad'  : 3,
            'ytick.major.size' : 2,
            'ytick.labelsize':10,
            'axes.labelsize':8,
            'font.size':8,
            'axes.labelpad':2,
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

import sys, os
import numpy as np
import pylab
import matplotlib.pyplot as plt
import pickle, pickle
import scipy.stats
from scipy.optimize import leastsq, curve_fit
import scipy
import statsmodels.api as sm # recommended import according to the docs
from sklearn.metrics import auc
import pandas as pd
from statsmodels.formula.api import ols
from copy import deepcopy
import matplotlib.pyplot as plt
import metrics_mod


save_type = 'pdf'
folder_name = './' #FOLDER WHERE NUMERICAL EXPERIMENTS ARE SAVED
save_folder = './' #FOLDER WHERE FIGURES ARE SAVED

try:
    arg = sys.argv
    ci_test = str(arg[1]) #par_corr
    variant = str(arg[2]) #To choose from "sample_size_highdegree" "highdim_highdegree"
                          # "tau_max_highdegree" "autocorr_highdegree"
except:
    arg = ''
    ci_test = 'par_corr'
    variant = 'sample_size_highdegree'

print(variant)
name = {'par_corr':r'ParCorr', 'gp_dc':r'GPDC'}

def f1score(precision_,recall_):
    return 2 * (precision_ * recall_) / (precision_ + recall_)

def get_metrics_from_file(para_setup):

    name_string = '%s-'*len(para_setup)  # % para_setup
    name_string = name_string[:-1]

    try:
        print("load from metrics file  %s_metrics.dat " % (folder_name + name_string % tuple(para_setup)))
        results = pickle.load(open(folder_name + name_string % tuple(para_setup) + '_metrics.dat', 'rb'), encoding='latin1')
    except Exception as e:
        print('failed from metrics file '  , tuple(para_setup))
        print(e)
        return None

    return results


def print_time(seconds, precision=1):
    if precision == 0:
        if seconds > 60*60.:
            return "%.0fh" % (seconds/3600.)
        elif seconds > 60.:
            return "%.0fmin" % (seconds/60.)
        else:
            return "%.0fs" % (seconds)
    else:
        if seconds > 60*60.:
            return "%.1fh" % (seconds/3600.)
        elif seconds > 60.:
            return "%.1fmin" % (seconds/60.)
        else:
            return "%.1fs" % (seconds)

def print_time_std(time, precision=1):

    mean = time.mean()
    std = time.std()
    if precision == 0:
        if mean > 60*60.:
            return r"%.0f$\pm$%.0fh" % (mean/3600., std/3600.)
        elif mean > 60.:
            return r"%.0f$\pm$%.0fmin" % (mean/60., std/60.)
        else:
            return r"%.0f$\pm$%.0fs" % (mean, std)
    else:
        if mean > 60*60.:
            return r"%.1f$\pm$%.1fh" % (mean/3600., std/3600.)
        elif mean > 60.:
            return r"%.1f$\pm$%.1fmin" % (mean/60., std/60.)
        else:
            return r"%.1f$\pm$%.1fs" % (mean, std)

def draw_it(paras, which):

    figsize = (4, 3)
    capsize = .5
    marker1 = 'o'
    marker2 = 's'
    marker3 = '+'
    alpha_marker = 1.

    params = { 
           'legend.fontsize': 5,
           'legend.handletextpad': .05,
           'lines.color':'black',
           'lines.linewidth':.5,
           'lines.markersize':2,
            'xtick.labelsize':6.5,
            'xtick.major.pad'  : 1, 
            'xtick.major.size' : 2,
            'ytick.major.pad'  : 1,
            'ytick.major.size' : 2,
            'ytick.labelsize':6.5,
            'axes.labelsize':8,
            'font.size':8,
            'axes.labelpad':2,
            'axes.spines.right' : False,
            'axes.spines.top' : False,
            }
    mpl.rcParams.update(params)
    fig = plt.figure(figsize=figsize,dpi=600)
    gs = fig.add_gridspec(1, 2)
    ax1a = fig.add_subplot(gs[0, 0])
    ax2a = fig.add_subplot(gs[0, 1])

    anova = {}
    auc_dict = {}
    fpr_dict = {}
    pr_df = pd.DataFrame(columns=['method', 'n_bs', 'param', 'pc_alpha','recall','precision','contemp_recall',
    'contemp_precision'])

    for method in methods:
        anova[method] = {}
        anova[method]['adj_anylink_tpr'] = []
        anova[method]['adj_anylink_fpr'] = []
        anova[method]['adj_lagged_tpr'] = []
        anova[method]['adj_lagged_fpr'] = []
        anova[method]['adj_contemp_tpr'] = []
        anova[method]['adj_contemp_fpr'] = []
        anova[method]['edgemarks_contemp_recall'] = []
        anova[method]['edgemarks_contemp_precision'] = []
        anova[method]['computation_time'] = []

        if "bootstrap" in method:
            n_bs_method = n_bs
        else: 
            n_bs_method= [0]


        for idx,n_bs_here in enumerate(n_bs_method):
            for para in paras:
                for pc_alpha in pc_alpha_here:
                    pc_alpha = np.format_float_positional(float(pc_alpha),trim='-')
                    if which == 'auto':
                        auto_here = para
                        N_here = N
                        tau_max_here = tau_max
                        frac_unobserved_here = frac_unobserved
                        T_here = T

                    elif which == 'N':
                        N_here = para
                        auto_here = auto
                        tau_max_here = tau_max
                        frac_unobserved_here = frac_unobserved
                        T_here = T

                    elif which == 'tau_max':
                        N_here = N
                        auto_here = auto
                        tau_max_here = para
                        frac_unobserved_here = frac_unobserved
                        T_here = T

                    elif which == 'sample_size':
                        N_here = N
                        auto_here = auto
                        tau_max_here = tau_max
                        frac_unobserved_here = frac_unobserved
                        T_here = para

                    n_links_here = links_from_N(N_here)

                   
                    para_setup = (model, N_here, n_links_here, min_coeff, coeff, auto_here, contemp_fraction, frac_unobserved_here,  
                                    max_true_lag, T_here, ci_test, method, pc_alpha, tau_max_here, n_bs_here) 
                    metrics_dict = get_metrics_from_file(para_setup)
                    if metrics_dict is not None:
                        current_res = pd.DataFrame([[
                                    method, n_bs_here, para, pc_alpha, metrics_dict['adj_anylink_recall'][0], metrics_dict['adj_anylink_precision'][0],
                                    metrics_dict['edgemarks_contemp_recall'][0],metrics_dict['edgemarks_contemp_precision'][0]
                                    ]], 
                                    columns=['method', 'n_bs', 'param', 'pc_alpha','recall','precision','contemp_recall','contemp_precision'])
                        pr_df = pd.concat([pr_df,current_res])

    #Find common recall coverage for all parameters settings (methods, varying_param)
    pc_alpha_min, pc_alpha_max = min(pc_alpha_here),max(pc_alpha_here)
    pc_alpha_min = np.format_float_positional(float(pc_alpha_min),trim='-')
    pc_alpha_max = np.format_float_positional(float(pc_alpha_max),trim='-')
    lower_common_recall = pr_df[pr_df.pc_alpha == pc_alpha_min].max().recall
    upper_common_recall = pr_df[pr_df.pc_alpha == pc_alpha_max].min().recall
    print(pr_df[pr_df.pc_alpha == pc_alpha_min])
    pr_df[pr_df.pc_alpha == pc_alpha_max]
    pc_alpha_contemp_min, pc_alpha_contemp_max = min(pc_alpha_here),0.2
    pc_alpha_contemp_min = np.format_float_positional(float(pc_alpha_contemp_min),trim='-')
    pc_alpha_contemp_max = np.format_float_positional(float(pc_alpha_contemp_max),trim='-')
    lower_common_contemp_recall = pr_df[pr_df.pc_alpha == pc_alpha_contemp_min].max().contemp_recall
    upper_common_contemp_recall = pr_df[pr_df.pc_alpha == pc_alpha_contemp_max].min().contemp_recall
    print(pr_df[pr_df.pc_alpha == pc_alpha_contemp_min])
    print(pr_df[pr_df.pc_alpha == pc_alpha_contemp_max])
    for method in methods:
        if "bootstrap" in method:
            n_bs_method = n_bs
        else: 
            n_bs_method= [0]
        for idx,n_bs_here in enumerate(n_bs_method):
            for para in paras:
                print(para)
                fx = []
                x=[]
                fx_c = []
                x_c=[]
                for pc_alpha in pc_alpha_here:
                    pc_alpha = np.format_float_positional(float(pc_alpha),trim='-')
                    data_query = pr_df.query('method==@method and n_bs==@n_bs_here and param==@para and pc_alpha == @pc_alpha')
                    x.append(data_query.recall.values[0])
                    fx.append(data_query.precision.values[0])
                    if float(pc_alpha) <= 0.1:
                        x_c.append(data_query.contemp_recall.values[0])
                        fx_c.append(data_query.contemp_precision.values[0])
                lower_interp_precision = np.interp(lower_common_recall,x,fx)
                upper_interp_precision = np.interp(upper_common_recall,x,fx)
                lower_interp_precision_c = np.interp(lower_common_contemp_recall,x_c,fx_c)
                upper_interp_precision_c = np.interp(upper_common_contemp_recall,x_c,fx_c)
                final_recall = [lower_common_recall]
                final_precision = [lower_interp_precision]
                final_recall_c = [lower_common_contemp_recall]
                final_precision_c = [lower_interp_precision_c]
                for ix in range(len(x)):
                    if upper_common_recall>x[ix]>lower_common_recall:
                        final_recall.append(x[ix])
                        final_precision.append(fx[ix])
                for ix_c in range(len(x_c)):
                    if upper_common_contemp_recall>x_c[ix_c]>lower_common_contemp_recall:
                        final_recall_c.append(x_c[ix_c])
                        final_precision_c.append(fx_c[ix_c])
                final_recall.append(upper_common_recall)
                final_precision.append(upper_interp_precision)
                final_recall_c.append(upper_common_contemp_recall)
                final_precision_c.append(upper_interp_precision_c)

                auc_value = auc(final_recall,final_precision)
                auc_c_value = auc(final_recall_c,final_precision_c)
                if "bootstrap" in method:
                    para_plot = paras.index(para) + (methods.index(method)+idx)/float(len(n_bs)+len(methods)-1)*.6
                else: 
                    para_plot = paras.index(para) + (methods.index(method)+idx)/float(len(n_bs)+len(methods)-1)*.6
                
                ax1a.errorbar(para_plot, auc_value, capsize=capsize,  alpha=alpha_marker,
                             color=color_picker(method,idx), marker=marker2, linestyle='dashed')
                ax2a.errorbar(para_plot, auc_c_value, capsize=capsize,  alpha=alpha_marker,
                             color=color_picker(method,idx), marker=marker2, linestyle='dashed')
                     
    axes = {'ax1a':ax1a, 'ax2a':ax2a}
    for axname in axes:

        ax = axes[axname]

        if which == 'N':
            ax.set_xlim(-0.5, len(paras))
            if ci_test == 'par_corr':
                ax.xaxis.set_ticks([paras.index(p) for p in paras] )
                ax.xaxis.set_ticklabels([str(p) for p in paras] )
            else:
                ax.xaxis.set_ticks([paras.index(p) for p in paras] )
                ax.xaxis.set_ticklabels([str(p) for p in paras] )

        elif which == 'auto':
            ax.xaxis.set_ticks([paras.index(p) for p in paras] )
            ax.xaxis.set_ticklabels([str(p) for p in paras] )
        elif which == 'pc_alpha':
            ax.set_xlim(0, len(paras))
            ax.xaxis.set_ticks([paras.index(p) for p in paras])
            ax.set_xticklabels([str(p) for p in paras], rotation=75, horizontalalignment="center")
            
        elif which == 'tau_max':
            ax.set_xlim(-0.5, len(paras))
            ax.xaxis.set_ticks([paras.index(p) for p in paras] )
            ax.xaxis.set_ticklabels([str(p) for p in paras] )

        elif which == 'unobserved':
            ax.set_xlim(0, len(paras))
            ax.xaxis.set_ticks([paras.index(p) for p in paras] )
            ax.xaxis.set_ticklabels([str(p) for p in paras] )

        elif which == 'sample_size':
            print(paras)
            ax.set_xlim(0, len(paras))
            ax.xaxis.set_ticks([paras.index(p) for p in paras] )
            ax.xaxis.set_ticklabels([str(p) for p in paras] )

        # ax.set_xlabel(xlabel, fontsize=8)
        for line in ax.get_lines():
            line.set_clip_on(False)
        # Disable spines.
        # if not 'ax3a' in axname:
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.spines['left'].set_position(('outward', 3))
        ax.spines['bottom'].set_position(('outward', 3))

        ax.grid(axis='y', linewidth=0.3)

        pad = 3     

        if axname == 'ax1a':
            ax.set_title('Adj. precision-recall AUC', fontsize=10, pad=pad)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
         
        elif axname == 'ax2a':
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.set_title('Contemp. \n precision-recall  AUC', fontsize=10, pad=pad)



    axlegend = fig.add_axes([0.0, .93, 1., .05])
    axlegend.axis('off')
    for method in methods:
        if "bootstrap" in method:
            for idx,n_bs_here in enumerate(n_bs):
                method_label ="Bagged("+str(n_bs_here)+")-PCMCI+"
                color_ = color_picker(method,idx) 
                
                axlegend.errorbar([], [], linestyle='',
                capsize=capsize, label=method_label,
                color= color_, marker='s')
        else:
            method_label ="PCMCI+"
            color_= color_picker(method) 
            axlegend.errorbar([], [], linestyle='',
            capsize=capsize, label=method_label,
            color=color_, marker='s')
    
    if not 'paper' in variant:
        ncol = 4
        fontsize = 10
    else:
        ncol = 4
        fontsize = 10
    axlegend.legend(ncol=ncol,
             loc='lower left',
            markerscale=3,
            columnspacing=.75,
            labelspacing=.01,
            fontsize=fontsize, framealpha=.5
            )
    if not 'paper' in variant:
        ncol = 5
        fontsize = 6
        ncol2 =3
        ncolroc=1
    else:
        ncol = 5
        fontsize = 6
        ncol2 =3
        ncolroc=1

    axlegend.legend(ncol=ncol,
             loc='lower left',
            markerscale=3,
            columnspacing=.75,
            labelspacing=.01,
            fontsize=fontsize, framealpha=.5
            )
    
    #produce letter mark top left
    if 'paper' in variant:
        if 'autocorr' in variant:
            fig.text(0., 0.93, "C", fontsize=12, fontweight='bold',
                ha='left', va='top')
        elif 'highdim' in variant:
            fig.text(0., 0.93, "A", fontsize=12, fontweight='bold',
                ha='left', va='top')
        elif 'sample_size' in variant:
            fig.text(0., 0.93, "B", fontsize=12, fontweight='bold',
                ha='left', va='top')
        elif 'tau_max' in variant:
            fig.text(0., 0.93, "D", fontsize=12, fontweight='bold',
                ha='left', va='top')

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

    if which == 'N':
        fig.text(0.5, 0., r"Number of variables $N$", fontsize=10,
            horizontalalignment='center', va='bottom')
        fig.text(1., 1., r"%s: $T=%d, a=%s$, " %(model_name, T, auto) 
                            + r"%s, $\tau_{\max}=%d$" %(name[ci_test],tau_max),
         fontsize=7.5, ha='right', va='top')
    elif which == 'auto':
        fig.text(0.5, 0., r"Autocorrelation $a$", fontsize=10,
            horizontalalignment='center', va='bottom')
        fig.text(1., 1., r"%s: $N=%d, T=%d$" %(model_name, N, T) 
                        + r", %s, $\tau_{\max}=%d$" %(name[ci_test], tau_max),
         fontsize=8, ha='right', va='top')

    elif which == 'tau_max':
        fig.text(0.5, 0., r"Time lag $\tau_{\max}$", fontsize=10,
            horizontalalignment='center', va='bottom')
        fig.text(1., 1., r"%s: $N=%d, T=%d, a=%s$" %(model_name, N, T, auto) 
                         + r", %s" %(name[ci_test]),
         fontsize=8, ha='right', va='top')
    elif which == 'unobserved':
        fig.text(0.5, 0., r"Frac. unobserved", fontsize=9,
            horizontalalignment='center', va='bottom')
        fig.text(1., 1., r"%s: $N=%d, T=%d, a=%s, \tau_{\max}=%d$" %(model_name, N, T, auto) 
                            +"\n" + r"%s, $\alpha=%s, \tau_{\max}=%d$" %(name[ci_test], pc_alpha, tau_max),
         fontsize=8, ha='right', va='top')
    elif which == 'sample_size':
        fig.text(0.5, 0., r"Sample size $T$", fontsize=10,
            horizontalalignment='center', va='bottom')
        fig.text(1., 1., r"%s: $N=%d, a=%s$" %(model_name, N, auto) 
                        + r", %s, $\tau_{\max}=%d$" %(name[ci_test], tau_max),
         fontsize=8, ha='right', va='top')
    elif which == 'pc_alpha':
        fig.text(0.5, 0., r"$\alpha$", fontsize=9,
            horizontalalignment='center', va='bottom')

        fig.text(1., 1., r"%s: $N=%d, T=%d, a=%s$" %(model_name, N, T, auto) 
                        + r", %s, $\tau_{\max}=%d$" %(name[ci_test], tau_max),
         fontsize=6, ha='right', va='top')
    fig.subplots_adjust(left=0.07, right=0.93, hspace=.3, bottom=0.12, top=0.85, wspace=.3)
    print("Saving plot to %s" %(save_folder + '%s.%s' %(save_suffix, save_type)))
    fig.savefig(save_folder + '%s.%s' %(save_suffix, save_type))
    plot_files.append(save_folder + '%s.%s' %(save_suffix, save_type))

def adjust_lightness(color, amount=0.5):
    #Function to create a color shading
    # amount <1 darkens the color, amount >1 lightens the color
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
        color_scaling = [adjust_lightness('green', amount=i) for i in [3.4,2.4,1,0]]
        return color_scaling[2]
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

    # if 'highdim' in variant and :
    #     return num_nodes
    if num_nodes == 2:
        return 1

    if 'fixeddensity' in model:
        return max(num_nodes, int(0.2*num_nodes*(num_nodes-1.)/2.))   # CONSTANT LINK DENSITY 0.2 FOR NON_TIME SERIES !!!
    elif 'highdegree' in model:
        return int(1.5*num_nodes) 
    else:
        return num_nodes



if __name__ == '__main__':

    
    save_type = 'pdf'
    paper = True
    fpr_precision = 'precision'

    if 'versions' in variant:
        methods = [
            'standard_pcmci+', 
            'bootstrap_pcmci+'
            ]
    else:
        methods = []

        methods += [
            'standard_pcmci+',
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

    if  'autocorr' in variant:
        if ci_test == 'par_corr':
            T_here = [500]
            N_here = [5]
            num_rows = 4
        else:
            T_here = [500]
            N_here = [5]
            num_rows = 3

        tau_max = 5
        # vary_auto = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]
        vary_auto = [ 0.4, 0.5, 0.6, 0.8, 0.9, 0.95, 0.98]#, 0.99, 0.999]#[0., 0.2,

        pc_alpha_here = [0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01,0.02,0.05,0.1,0.2,0.3,
                                    0.4, 0.5, 0.6, 0.8, 0.9, 0.999] #
        
        min_coeff = 0.1
        coeff = 0.5      
        frac_unobserved = 0.
        contemp_fraction = 0.3
        max_true_lag = 5
        # T = 500
        n_bs= [100]

        pdfjam_suffix = variant + "_" + str(ci_test)
        for T in T_here: 
         for N in N_here:
            n_links = links_from_N(N)  
            para_setup_name = (variant, N, n_links, min_coeff, coeff, contemp_fraction, frac_unobserved,  
                            max_true_lag, T, ci_test, "auc", tau_max,n_bs) 

            save_suffix = '%s-'*len(para_setup_name) % para_setup_name
            save_suffix = save_suffix[:-1]

            print(save_suffix)
            draw_it(paras=vary_auto, which='auto')  

            if test:
                sys.exit(0)

    elif 'highdim' in variant:
        if ci_test == 'par_corr':
            T_here = [500]
            vary_N =   [3,5,10,]
            auto_here = [0.95] 
            num_rows = 4
        else:
            T_here = [500] #, 1000]
            vary_N =  [2,3,5,8,10,15]  
            auto_here = [0.95] 
            num_rows = 4

        contemp_fraction = 0.3
        frac_unobserved = 0.
        max_true_lag = 5
        tau_max = 5

        min_coeff = 0.1
        coeff = 0.5
        pc_alpha_here = [0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01,0.02,0.05,0.1,0.2,0.3,
                                    0.4, 0.5, 0.6, 0.8, ]
        n_bs = [100]

        pdfjam_suffix = variant + "_" + str(ci_test)
        
        for T in T_here:
            for auto in auto_here:
                para_setup_name = (variant, min_coeff, coeff, auto, contemp_fraction, frac_unobserved, max_true_lag, T, 
                                        ci_test, 'auc', tau_max, n_bs)
                save_suffix = '%s-'*len(para_setup_name) % para_setup_name
                save_suffix = save_suffix[:-1]
                draw_it(paras=vary_N, which='N')  

                if test:
                    sys.exit(0)

    elif  'sample_size' in variant:

        if ci_test == 'par_corr':
            vary_T = [100,200,500,1000]
            N_here = [5]
            auto_here = [0.95]
            num_rows = 4
        else:
            vary_T = [200, 500]
            N_here = [5, 10]
            auto_here = [0., 0.5, 0.9, 0.95]  
            num_rows = 4

        min_coeff = 0.1
        coeff = 0.5     
        pc_alpha_here = [0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01,0.02,0.05,0.1,0.2,0.3,
                                    0.4, 0.5, 0.6, 0.8, 0.9, 0.999]
        contemp_fraction = 0.3
        frac_unobserved = 0.

        max_true_lag = 5
        tau_max = 5
        pdfjam_suffix = variant + "_" + str(ci_test)
        n_bs = [100]
        for N in N_here:
          n_links = links_from_N(N)   
          for auto in auto_here:
                para_setup_name = (variant, N, n_links, min_coeff, coeff, contemp_fraction, frac_unobserved, 
                                    max_true_lag, auto, ci_test, "auc", tau_max, n_bs)
                save_suffix = '%s-'*len(para_setup_name) % para_setup_name
                save_suffix = save_suffix[:-1]

                print(save_suffix)
                draw_it(paras=vary_T, which='sample_size')  

                if test:
                    sys.exit(0)
    if 'tau_max' in variant:

        if ci_test == 'par_corr':
            T_here = [500]
            N_here = [5]
            auto_here = [0.95] 
            # vary_N =  [5, 10, 20, 30, 40]
            num_rows = 4
            vary_tau_max = [5, 10, 15, 20, 25, 30, 35, 40]

        else:
            T_here = [200, 500]
            N_here = [5]
            auto_here = [0., 0.5, 0.9, 0.95] 
            num_rows = 4
            vary_tau_max = [5, 10, 15, 20]

        min_coeff = 0.1
        coeff = 0.5       
        pc_alpha_here = [0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01,0.02,0.05,0.1,0.2,0.3,
                                    0.4, 0.5, 0.6]
 
        contemp_fraction = 0.3
        frac_unobserved = 0.
        max_true_lag = 5
        n_bs = [100]
        pdfjam_suffix = variant + "_" + str(ci_test)

        for T in T_here:

         for N in N_here:
          n_links = links_from_N(N)   
          for auto in auto_here:
                para_setup_name = (variant, N, n_links, min_coeff, coeff, contemp_fraction, frac_unobserved,
                                    max_true_lag, auto, T, ci_test, "auc", n_bs)
                save_suffix = '%s-'*len(para_setup_name) % para_setup_name
                save_suffix = save_suffix[:-1]

                print(save_suffix)
                draw_it(paras=vary_tau_max, which='tau_max')
