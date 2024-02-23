import matplotlib as mpl
from matplotlib.ticker import PercentFormatter
import sys
import numpy as np
import pylab
import matplotlib.pyplot as plt
import pickle, pickle
import pandas as pd
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt

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
    samples_int = int(samples)
    if np.linalg.matrix_rank(data1) == 1:
        data1 = data1.reshape(samples_int,1)
        data2 = data2.reshape(samples_int,1)
    auroc = 1. - (np.vstack((data1,data2)).argsort(axis=0).argsort(axis=0)[:samples_int].sum(axis=0) - samples*(samples+1.)/2.)/samples**2
    auroc[np.any(np.isnan(data1), axis=0)*np.any(np.isnan(data2), axis=0)] = np.nan
    return auroc



params = { 'figure.figsize': (8, 10),
           'legend.fontsize': 8,
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
            'axes.labelpad':2,
            'pdf.fonttype' : 42,
            'figure.dpi': 600,
            }

save_type = 'pdf'
folder_name = './' #FOLDER WHERE NUMERICAL EXPERIMENTS ARE SAVED
save_folder = './' #FOLDER WHERE FIGURES ARE SAVED

try:
    arg = sys.argv
    ci_test = str(arg[1])  #par_corr or gp_dc
    variant = str(arg[2])  #sample_size/tau_max/highdim/autocorr/pc_alpha + highdegree/"" + mixed/nonlinear/""
except:
    arg = ''
    ci_test = 'par_corr'
    variant = 'autocorr_highdegree'
print(variant)
name = {'par_corr':r'ParCorr', 'gp_dc':r'GPDC'}

def f1score(precision_,recall_):
    return 2 * (precision_ * recall_) / (precision_ + recall_)

def get_metrics_from_file(para_setup):

    name_string = '%s-'*len(para_setup)
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
            'xtick.labelsize':5,
            'xtick.major.pad'  : 1, 
            'xtick.major.size' : 2,
            'ytick.major.pad'  : 1,
            'ytick.major.size' : 2,
            'ytick.labelsize':5,
            'axes.labelsize':8,
            'font.size':8,
            'axes.labelpad':2,
            'axes.spines.right' : False,
            'axes.spines.top' : False,
            }
    mpl.rcParams.update(params)

    fig = plt.figure(figsize=figsize,dpi=600)
    fig2 = plt.figure(figsize=(1.5,2),dpi=600)
    gs = fig.add_gridspec(3, 3)

    axfig2 = fig2.add_subplot(111)
    ax1a = fig.add_subplot(gs[0, 0])
    ax1b = fig.add_subplot(gs[1, 0])
    ax1c = fig.add_subplot(gs[2, 0])
    ax2a = fig.add_subplot(gs[0, 1])
    ax2b = fig.add_subplot(gs[1, 1])
    ax2c = fig.add_subplot(gs[2, 1])
    ax3a = fig.add_subplot(gs[0, 2])
    ax3b = fig.add_subplot(gs[1, 2])

    if which == 'pc_alpha':
        figroc = plt.figure(figsize=(3,2),dpi=600)
        axroc_anylink = figroc.add_subplot(121)
        axroc_contemp = figroc.add_subplot(122)
        figalpha = plt.figure(figsize=(2,2),dpi=600)
    if fpr_precision == 'fpr':
        if which == 'pc_alpha':
            print(paras)
            ax1b.plot([paras.index(para) for para in paras], paras, color='grey', linewidth=2.)
        else:    
            ax1b.axhline(pc_alpha, color='grey', linewidth=2.)
    if which == 'pc_alpha':
        axfig2.plot([paras.index(para) for para in paras], paras, color='grey', linewidth=2.)
    else:
        axfig2.axhline(pc_alpha, color='grey', linewidth=2.)

    anova = {}
    auc_dict = {}
    fpr_dict = {}
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
            if which == "pc_alpha":
                recall_list_method = []
                precision_list_method = []
                contemp_recall_list_method = []
                contemp_precision_list_method = []
            for para in paras:
                if which == 'auto':
                    auto_here = para
                    N_here = N
                    tau_max_here = tau_max
                    frac_unobserved_here = frac_unobserved
                    pc_alpha_here = pc_alpha
                    T_here = T

                elif which == 'N':
                    N_here = para
                    auto_here = auto
                    tau_max_here = tau_max
                    frac_unobserved_here = frac_unobserved
                    pc_alpha_here = pc_alpha
                    T_here = T

                elif which == 'tau_max':
                    N_here = N
                    auto_here = auto
                    tau_max_here = para
                    frac_unobserved_here = frac_unobserved
                    pc_alpha_here = pc_alpha
                    T_here = T

                elif which == 'sample_size':
                    N_here = N
                    auto_here = auto
                    tau_max_here = tau_max
                    frac_unobserved_here = frac_unobserved
                    pc_alpha_here = pc_alpha
                    T_here = para

                elif which == 'unobserved':
                    N_here = N
                    auto_here = auto
                    tau_max_here = tau_max
                    frac_unobserved_here = para
                    pc_alpha_here = pc_alpha
                    T_here = T

                elif which == 'pc_alpha':
                    N_here = N
                    auto_here = auto
                    tau_max_here = tau_max
                    frac_unobserved_here = frac_unobserved
                    if para== 0. or para == 1.:
                        pc_alpha_here=para
                    else:
                        pc_alpha_here = np.format_float_positional(np.float64(para),trim='-')
                    T_here = T

                n_links_here = links_from_N(N_here)

                if "bootstrap" in method:
                    para_plot = paras.index(para) + (methods.index(method)+idx)/float(len(n_bs)+len(methods)-1)*.6
                else: 
                    para_plot = paras.index(para) + (methods.index(method)+idx)/float(len(n_bs)+len(methods)-1)*.6
                if aggregation =="alternative" and "bootstrap" in method:
                    para_setup = (model, N_here, n_links_here, min_coeff, coeff, auto_here, contemp_fraction, frac_unobserved_here,  
                                    max_true_lag, T_here, ci_test, method, pc_alpha_here, tau_max_here, n_bs_here,aggregation)
                else:
                    para_setup = (model, N_here, n_links_here, min_coeff, coeff, auto_here, contemp_fraction, frac_unobserved_here,  
                                    max_true_lag, T_here, ci_test, method, pc_alpha_here, tau_max_here, n_bs_here)
                    
                metrics_dict = get_metrics_from_file(para_setup)
                if metrics_dict is not None:

                    anova[method]['adj_anylink_tpr'].append((para, metrics_dict['adj_anylink_tpr']))
                    anova[method]['adj_anylink_fpr'].append((para, metrics_dict['adj_anylink_fpr']))
                    anova[method]['adj_lagged_tpr'].append((para, metrics_dict['adj_lagged_tpr']))
                    anova[method]['adj_lagged_fpr'].append((para, metrics_dict['adj_lagged_fpr']))
                    anova[method]['adj_contemp_tpr'].append((para, metrics_dict['adj_contemp_tpr']))
                    anova[method]['adj_contemp_fpr'].append((para, metrics_dict['adj_contemp_fpr']))
                    anova[method]['edgemarks_contemp_recall'].append((para, metrics_dict['edgemarks_contemp_recall']))
                    anova[method]['edgemarks_contemp_precision'].append((para, metrics_dict['edgemarks_contemp_precision']))
                    anova[method]['computation_time'].append((para, metrics_dict['computation_time']))

                    if which == "pc_alpha":
                        recall_list_method.append(metrics_dict['adj_anylink_recall'][0])
                        precision_list_method.append(metrics_dict['adj_anylink_precision'][0])
                        if para <0.1:
                            contemp_recall_list_method.append(metrics_dict['edgemarks_contemp_recall'][0])
                            contemp_precision_list_method.append(metrics_dict['edgemarks_contemp_precision'][0])
                        fpr_dict.setdefault(method+str(n_bs_here),[])
                        fpr_dict[method+str(n_bs_here)].append(metrics_dict['adj_anylink_fpr'][0])
                    axfig2.errorbar(para_plot, *metrics_dict['adj_lagged_fpr'], capsize=capsize, alpha=alpha_marker,
                        color=color_picker(method,idx), marker=marker1, linestyle='solid')
                    axfig2.errorbar(para_plot, *metrics_dict['adj_auto_fpr'], capsize=capsize, alpha=alpha_marker,
                        color=color_picker(method,idx), marker=marker3, linestyle='solid')
                    axfig2.errorbar(para_plot, *metrics_dict['adj_contemp_fpr'], capsize=capsize, alpha=alpha_marker,
                        color=color_picker(method,idx), marker=marker2, linestyle='dashed')

                    ax1a.errorbar(para_plot, *metrics_dict['adj_lagged_recall'], capsize=capsize, alpha=alpha_marker,
                        color=color_picker(method,idx), marker=marker1, linestyle='solid')
                    ax1a.errorbar(para_plot, *metrics_dict['adj_auto_recall'], capsize=capsize, alpha=alpha_marker,
                        color=color_picker(method,idx), marker=marker3, linestyle='solid')
                    ax1a.errorbar(para_plot, *metrics_dict['adj_contemp_recall'], capsize=capsize, alpha=alpha_marker,
                        color=color_picker(method,idx), marker=marker2, linestyle='dashed')

                    ax1b.errorbar(para_plot, *metrics_dict['adj_lagged_%s' % fpr_precision], capsize=capsize, alpha=alpha_marker,
                        color=color_picker(method,idx), marker=marker1, linestyle='solid')
                    ax1b.errorbar(para_plot, *metrics_dict['adj_auto_%s' % fpr_precision], capsize=capsize, alpha=alpha_marker,
                        color=color_picker(method,idx), marker=marker3, linestyle='solid')
                    ax1b.errorbar(para_plot, *metrics_dict['adj_contemp_%s' % fpr_precision], capsize=capsize,  alpha=alpha_marker,
                        color=color_picker(method,idx), marker=marker2, linestyle='dashed')

                    f1_adj_lagged = f1score(metrics_dict['adj_lagged_%s' % fpr_precision][0],metrics_dict['adj_lagged_recall'][0])
                    ax1c.errorbar(para_plot, f1_adj_lagged, capsize=capsize,  alpha=alpha_marker,
                        color=color_picker(method,idx), marker=marker1, linestyle='dashed')
                    f1_adj_auto = f1score(metrics_dict['adj_auto_%s' % fpr_precision][0],metrics_dict['adj_auto_recall'][0])
                    ax1c.errorbar(para_plot, f1_adj_auto, capsize=capsize,  alpha=alpha_marker,
                        color=color_picker(method,idx), marker=marker3, linestyle='dashed')
                    f1_adj_contemp = f1score(metrics_dict['adj_contemp_%s' % fpr_precision][0],metrics_dict['adj_contemp_recall'][0])
                    ax1c.errorbar(para_plot, f1_adj_contemp, capsize=capsize,  alpha=alpha_marker,
                        color=color_picker(method,idx), marker=marker2, linestyle='dashed')

                    ax2a.errorbar(para_plot, *metrics_dict['edgemarks_contemp_recall'], capsize=capsize,  alpha=alpha_marker,
                        color=color_picker(method,idx), marker=marker2, linestyle='dashed')

                    ax2b.errorbar(para_plot, *metrics_dict['edgemarks_contemp_precision'], capsize=capsize, alpha=alpha_marker,
                        color=color_picker(method,idx), marker=marker2, linestyle='dashed')

                    f1_edgemarks = f1score(metrics_dict['edgemarks_contemp_precision'][0],metrics_dict['edgemarks_contemp_recall'][0])
                    ax2c.errorbar(para_plot, f1_edgemarks, capsize=capsize, alpha=alpha_marker,
                        color=color_picker(method,idx), marker=marker2, linestyle='dashed')

                    ax3b.errorbar(para_plot, *metrics_dict['conflicts_contemp'], capsize=capsize,  alpha=alpha_marker,
                        color=color_picker(method,idx), marker=marker2)

                    ax3a.errorbar(para_plot, metrics_dict['computation_time'][0], metrics_dict['computation_time'][1].reshape(2, 1), capsize=capsize, alpha=alpha_marker,
                        color=color_picker(method,idx), marker='p', linestyle='solid')
            if which == "pc_alpha":
                axroc_anylink.plot(recall_list_method,precision_list_method,color=color_picker(method,idx),marker=marker1, markersize=1,
                            alpha=alpha_marker, linewidth = 0.5, linestyle="solid")
                auc_dict[method+str(n_bs_here)+'_anylink'] = np.trapz(precision_list_method,recall_list_method)
                axroc_contemp.plot(contemp_recall_list_method,contemp_precision_list_method,color=color_picker(method,idx),marker=marker1, markersize=1,
                            alpha=alpha_marker, linewidth = 0.5, linestyle="solid")
                auc_dict[method+str(n_bs_here)+'_contemp'] = np.trapz(contemp_precision_list_method,contemp_recall_list_method)

    if run_anova:

        anova_results = {}
        for method in methods:
            anova_results[method] = {}
            for metric in anova[method].keys():
                anova_results[method][metric] = {}
                aslist = []
                for para, data in anova[method][metric]:
                    mean, std = data
                    val = mean
                    aslist.append((val, para))
                try:
                    df = pd.DataFrame.from_records(aslist, columns = [metric, 'para'])
                    results = ols('%s ~  C(para)' % metric, data=df).fit()       
                    coeffs = np.array(results.params)
                    pvals = np.array(results.pvalues)
                    ref_coeff = coeffs[0]
                    normalized_coeffs = coeffs[1:] / (np.array(paras[1:]) - paras[0])
                    if ref_coeff < 0.: ref_coeff = 0.
                    anova_results[method][metric]['ref_coeff'] = ref_coeff
                    anova_results[method][metric]['normalized_coeffs_mean'] = normalized_coeffs.mean()
                    anova_results[method][metric]['any_sig'] = len(np.where(pvals[1:] <= 0.01)[0])
                except Exception as e:
                    print("Something went wrong in ANOVA")
                    print(e)

    def draw_anova(axis, metric_here, loc='top'):
        for method in methods:
            if len(anova_results[method][metric_here]) > 0 and np.isnan(anova_results[method][metric_here]['normalized_coeffs_mean'])==False:
                if which == 'sample_size':
                    factor = 100.
                elif which == 'highdim':
                    factor = 10.   
                elif which == 'tau_max':
                    factor = 10.  
                else: factor = 1.
                if metric_here == 'computation_time':
                    string = r"{:.2f}{:+.2f}".format(anova_results[method][metric_here]['ref_coeff'], 
                        factor*anova_results[method][metric_here]['normalized_coeffs_mean'])
                elif metric_here == 'adj_lagged_fpr':
                    string = r"{:.1f}{:+.1f}".format(100.*anova_results[method][metric_here]['ref_coeff'], 
                        100.*factor*anova_results[method][metric_here]['normalized_coeffs_mean'])
                else:
                    percent_factor = 100.
                    string = r"{:d}{:+d}".format(
                        int(percent_factor*anova_results[method][metric_here]['ref_coeff']), 
                        int(percent_factor*factor*anova_results[method][metric_here]['normalized_coeffs_mean']))
                if anova_results[method][metric_here]['any_sig'] > 0:
                    string += r"$^*$/{}".format(int(factor))
                else:
                    string += r"/{}".format(int(factor))
                if loc == 'top':
                    y_loc = 1. - list(np.array(methods)[::-1]).index(method)/float(len(methods))*0.35
                elif loc == 'bottom':
                    y_loc = .4 - list(np.array(methods)[::-1]).index(method)/float(len(methods))*0.35
                axis.text(0., y_loc, string,
                        fontsize=5,
                        color = color_picker(method),
                        horizontalalignment='left',
                        verticalalignment='top',
                        weight='bold',
                        bbox=dict(boxstyle='square,pad=0', facecolor='white', linewidth=0., alpha=0.5),
                        transform=ax.transAxes)
    if which =="pc_alpha":
        axroc_anylink.set_xlim(min(recall_list_method)-0.01,1.+0.01)
        axroc_anylink.set_ylim(0,1.)
        axroc_anylink.grid(axis='y', linewidth=0.3)
        axroc_anylink.set_title('Adj. precision-recall Curve', fontsize=5, pad=2)
        axroc_anylink.set_xlabel('Adj. recall',fontsize=5)
        axroc_anylink.set_ylabel('Adj. precision',fontsize=5)
        auc_noskill=n_links_here/(N_here*N_here*(tau_max_here+1)-N_here)
        axroc_anylink.axhline(auc_noskill, color='blue', linestyle= 'dashed', linewidth=0.5)

        axroc_contemp.set_xlim(min(contemp_recall_list_method)-0.01,max(contemp_recall_list_method)+0.01)
        axroc_contemp.set_ylim(0,1.)
        axroc_contemp.grid(axis='y', linewidth=0.3)
        axroc_contemp.set_title('Contemp. orient. precision-recall Curve', fontsize=5, pad=2)
        axroc_contemp.set_xlabel('Contemp. orient. recall',fontsize=5)
        axroc_contemp.set_ylabel('Contemp. orient. precision',fontsize=5)

    axes = {'ax1a':ax1a, 'ax1b':ax1b, 'ax1c':ax1c, 'ax2a':ax2a, 'ax2b':ax2b, 'ax2c': ax2c, 
            'ax3a':ax3a, 'ax3b':ax3b, "axfig2":axfig2} #, 'ax4a':ax4a} #, 'ax4b':ax4b}
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
            ax.set_xlim(0, len(paras))
            ax.xaxis.set_ticks([paras.index(p) for p in np.array(paras)[::2]] )
            ax.xaxis.set_ticklabels([str(p) for p in np.array(paras)[::2]] )

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
            ax.set_xlim(0, len(paras))
            ax.xaxis.set_ticks([paras.index(p) for p in paras] )
            ax.xaxis.set_ticklabels([str(p) for p in paras] )

        # ax.set_xlabel(xlabel, fontsize=8)
        for line in ax.get_lines():
            line.set_clip_on(False)

        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.spines['left'].set_position(('outward', 3))
        ax.spines['bottom'].set_position(('outward', 3))

        ax.grid(axis='y', linewidth=0.3)

        pad = 2

        if axname == 'ax3b':
            label_1 = "Lagged"
            label_2 = "Contemp."
            label_3 = "Auto."

            ax.errorbar([], [], linestyle='',
                capsize=capsize, label=label_1,
                color='black', marker=marker1)
            ax.errorbar([], [], linestyle='',
                capsize=capsize, label=label_2,
                color='black', marker=marker2)
            ax.errorbar([], [], linestyle='',
                capsize=capsize, label=label_3,
                color='black', marker=marker3)
            ax.legend(ncol=2,
                    columnspacing=0.,
                    # bbox_to_anchor=(0., 1.02, 1., .03), borderaxespad=0, mode="expand", 
                    loc='upper left', fontsize=5, framealpha=0.3
                    ) #.draw_frame(False)
        

        if axname == 'ax1a':
            ax.set_title('Adj. Recall', fontsize=6, pad=pad)
            ax.set_ylim(0., 1.)
            ax.tick_params(labelbottom=False)    
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))

            if run_anova:
                draw_anova(ax, 'adj_lagged_tpr', 'bottom')

        elif axname == 'ax1b':
            if fpr_precision == 'precision':
                ax.set_title('Adj. precision', fontsize=6, pad=pad)
                ax.set_ylim(0., 1.)
                ax.tick_params(labelbottom=False)  
                ax.yaxis.set_major_formatter(PercentFormatter(1.0))
            else:
                ax.set_title('Adj. FPR', fontsize=6, pad=pad)
                ax.tick_params(labelbottom=False)  
                if which != 'pc_alpha':
                    ax.set_yscale('symlog', linthresh=pc_alpha*2)
                    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
                ax.set_ylim(0., 1.)
            if run_anova:
                draw_anova(ax, 'adj_lagged_fpr')

        elif axname == 'ax1c':
            if fpr_precision == 'precision':
                ax.set_title('Adj. F1-score', fontsize=6, pad=pad)
                ax.set_ylim(0., 1.)
                ax.yaxis.set_major_formatter(PercentFormatter(1.0))

        elif axname == 'axfig2':
            ax.set_title('Adj. FPR', fontsize=6, pad=pad)
            if which != 'pc_alpha':
                ax.yaxis.set_major_formatter(PercentFormatter(1.0))
                ax.set_ylim(0.,2*pc_alpha_here)
            else:
                ax.set_ylim(0.,paras[-1])

            

        elif axname == 'ax2a':
            ax.set_title('Contemp. orient. recall', fontsize=6, pad=pad)
            ax.set_ylim(0., 1.)
            ax.tick_params(labelbottom=False)    
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
   
            if run_anova:
                draw_anova(ax, 'edgemarks_contemp_recall')


        elif axname == 'ax2b':
            ax.set_title('Contemp. orient. precision', fontsize=6, pad=pad)
            ax.set_ylim(0., 1.)
            ax.tick_params(labelbottom=False)    
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))

            if run_anova:
                draw_anova(ax, 'edgemarks_contemp_precision')

        elif axname == 'ax2c':
            ax.set_title('Contemp. orient. F1-score', fontsize=6, pad=pad)
            ax.set_ylim(0., 1.)   
            ax.yaxis.set_major_formatter(PercentFormatter(1.0)) 


        elif axname == 'ax3b':
            ax.set_title('Conflicts', fontsize=6, pad=pad)
            ax.set_ylim(0., 1.)
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))


        elif axname == 'ax3a':
            ax.set_title('Runtime [s]', fontsize=6, pad=pad)
            # ax.set_ylim(0., 1.)
            ax.tick_params(labelbottom=False)    
            ax.set_yscale('log')
            if run_anova:
                draw_anova(ax, 'computation_time')

    axlegend = fig.add_axes([0.05, .89, 1., .05])
    axlegend.axis('off')
    for method in methods:
        if "bootstrap" in method:
            for idx,n_bs_here in enumerate(n_bs):
                method_label ="Bagged("+str(n_bs_here)+")"
                color_ = color_picker(method,idx) 
                
                axlegend.errorbar([], [], linestyle='',
                capsize=capsize, label=method_label,
                color= color_, marker='s')
        else:
            method_label ="PC"
            color_= color_picker(method) 
            axlegend.errorbar([], [], linestyle='',
            capsize=capsize, label=method_label,
            color=color_, marker='s')
    
    if not 'paper' in variant:
        ncol = 4
        fontsize = 6
    else:
        ncol = 4
        fontsize = 6
    axlegend.legend(ncol=ncol,
             loc='lower left',
            markerscale=3,
            columnspacing=.75,
            labelspacing=.01,
            fontsize=fontsize, framealpha=.5
            )

    axlegend2 = fig2.add_axes([0.05, .85, 1., .05])
    axlegend2.axis('off')
    for method in methods:
        if "bootstrap" in method:
            for idx,n_bs_here in enumerate(n_bs):
                method_label ="Bagged("+str(n_bs_here)+")"
                color_ = color_picker(method,idx) 
                
                axlegend2.errorbar([], [], linestyle='',
                capsize=capsize, label=method_label,
                color= color_, marker='s')
        else: 
            method_label ="PC"
            color_= color_picker(method) 
            axlegend2.errorbar([], [], linestyle='',
            capsize=capsize, label=method_label,
            color=color_, marker='s')

    if which == "pc_alpha":
        axlegend_roc = figroc.add_axes([0.1, .15, 0.4, .4])
        axlegend_roc.axis('off')
        for method in methods:
            if "bootstrap" in method:
                for idx,n_bs_here in enumerate(n_bs):
                    auc = auc_dict[method+str(n_bs_here)+"_anylink"]
                    auc_string= " (AUC: %.3f)"%auc
                    method_label ="Bagged("+str(n_bs_here)+")-PC"
                    color_ = color_picker(method,idx) 
                    axlegend_roc.errorbar([], [], linestyle='',
                    capsize=capsize, label=method_label,
                    color= color_, marker='s')
            else: 
                auc = auc_dict[method+"0"+"_anylink"]
                auc_string= " (AUC: %.3f)"%auc
                method_label ="PC"
                color_= color_picker(method) 
                axlegend_roc.errorbar([], [], linestyle='',
                capsize=capsize, label=method_label,
                color=color_, marker='s')
        axlegend_roc.errorbar([],[],linestyle='dashed',color="blue",linewidth=0.5,label="No skill", marker="")
        
        axlegend_roc2 = figroc.add_axes([0.55, .15, 0.4, .4])
        axlegend_roc2.axis('off')
        for method in methods:
            if "bootstrap" in method:
                for idx,n_bs_here in enumerate(n_bs):
                    auc = auc_dict[method+str(n_bs_here)+"_contemp"]
                    auc_string= " (AUC: %.3f)"%auc
                    method_label ="Bagged("+str(n_bs_here)+")-PC"
                    color_ = color_picker(method,idx) 
                    axlegend_roc2.errorbar([], [], linestyle='',
                    capsize=capsize, label=method_label,
                    color= color_, marker='s')
            else: 
                auc = auc_dict[method+"0"+"_contemp"]
                auc_string= " (AUC: %.3f)"%auc
                method_label ="PC"
                color_= color_picker(method) 
                axlegend_roc2.errorbar([], [], linestyle='',
                capsize=capsize, label=method_label,
                color=color_, marker='s')

    if not 'paper' in variant:
        ncol = 5
        fontsize = 5
        ncol2 =3
        ncolroc=1
    else:
        ncol = 5
        fontsize = 5
        ncol2 =3
        ncolroc=1


    axlegend2.legend(ncol=ncol2,
             loc='lower left',
            markerscale=1.5,
            columnspacing=.15,
            labelspacing=.45,
            fontsize=3, framealpha=.5
            )

    axlegend.legend(ncol=ncol,
             loc='lower left',
            markerscale=3,
            columnspacing=.75,
            labelspacing=.01,
            fontsize=fontsize, framealpha=.5
            )

    if which == "pc_alpha":
        axlegend_roc.legend(ncol=ncolroc,
                loc='lower left',
                markerscale=1.5,
                columnspacing=.15,
                labelspacing=.45,
                fontsize=4, framealpha=.5
                )
        axlegend_roc2.legend(ncol=ncolroc,
                loc='lower left',
                markerscale=1.5,
                columnspacing=.15,
                labelspacing=.45,
                fontsize=4, framealpha=.5
                )

    if 'paper' in variant and SM is False:
        if 'autocorr' in variant:  # and ci_test == 'par_corr':
            fig.text(0., 1., "A", fontsize=12, fontweight='bold',
                ha='left', va='top')
            fig2.text(0., 1., "A", fontsize=12, fontweight='bold',
                ha='left', va='top')
        elif 'highdim' in variant:
            fig.text(0., 1., "B", fontsize=12, fontweight='bold',
                ha='left', va='top')
            fig2.text(0., 1., "B", fontsize=12, fontweight='bold',
                ha='left', va='top')
        elif 'sample_size' in variant:
            fig.text(0., 1., "C", fontsize=12, fontweight='bold',
                ha='left', va='top')
            fig2.text(0., 1., "C", fontsize=12, fontweight='bold',
                ha='left', va='top')
        elif 'tau_max' in variant:
            fig.text(0., 1., "D", fontsize=12, fontweight='bold',
                ha='left', va='top')
            fig2.text(0., 1., "D", fontsize=12, fontweight='bold',
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
        fig.text(0.5, 0., r"Number of variables $N$", fontsize=8,
            horizontalalignment='center', va='bottom')

        fig.text(1., 1., r"%s: $T=%d, a=%s$" %(model_name, T, auto) 
                            +"\n" + r"%s, $\alpha=%s, \tau_{\max}=%d$" %(name[ci_test],pc_alpha,tau_max),
         fontsize=6, ha='right', va='top')
        fig2.text(0.5, 0., r"Number of variables $N$", fontsize=5,
            horizontalalignment='center', va='bottom')

        fig2.text(1., 1., r"%s: $T=%d, a=%s$, %s" %(model_name, T, auto, name[ci_test]) 
                            +"\n" + r"$\alpha=%s, \tau_{\max}=%d$" %(pc_alpha,tau_max),
         fontsize=4, ha='right', va='top')
    elif which == 'auto':
        fig.text(0.5, 0., r"Autocorrelation $a$", fontsize=8,
            horizontalalignment='center', va='bottom')
        fig.text(1., 1., r"%s: $N=%d, T=%d$" %(model_name, N, T) 
                            +"\n" + r"%s, $\alpha=%s, \tau_{\max}=%d$" %(name[ci_test], pc_alpha, tau_max),
         fontsize=6, ha='right', va='top')
        fig2.text(0.5, 0., r"Autocorrelation $a$", fontsize=5,
            horizontalalignment='center', va='bottom')
        fig2.text(1., 1., r"%s: $N=%d, T=%d$, %s" %(model_name, N, T, name[ci_test]) 
                            +"\n" + r"$\alpha=%s, \tau_{\max}=%d$" %( pc_alpha, tau_max),
         fontsize=4, ha='right', va='top')
    elif which == 'tau_max':
        fig.text(0.5, 0., r"Time lag $\tau_{\max}$", fontsize=8,
            horizontalalignment='center', va='bottom')
        fig.text(1., 1., r"%s: $N=%d, T=%d, a=%s$" %(model_name, N, T, auto) 
                            +"\n" + r"%s, $\alpha=%s$" %(name[ci_test], pc_alpha),
         fontsize=6, ha='right', va='top')
        fig2.text(0.5, 0., r"Time lag $\tau_{\max}$", fontsize=5,
            horizontalalignment='center', va='bottom')

        fig2.text(1., 1., r"%s: $N=%d, T=%d, a=%s$" %(model_name, N, T, auto) 
                            +"\n" + r"%s, $\alpha=%s$" %(name[ci_test], pc_alpha),
         fontsize=4, ha='right', va='top')
    elif which == 'unobserved':
        fig.text(0.5, 0., r"Frac. unobserved", fontsize=8,
            horizontalalignment='center', va='bottom')
        fig.text(1., 1., r"%s: $N=%d, T=%d, a=%s, \tau_{\max}=%d$" %(model_name, N, T, auto) 
                            +"\n" + r"%s, $\alpha=%s, \tau_{\max}=%d$" %(name[ci_test], pc_alpha, tau_max),
         fontsize=6, ha='right', va='top')
        fig2.text(0.5, 0., r"Frac. unobserved", fontsize=5,
            horizontalalignment='center', va='bottom')
        fig2.text(1., 1., r"%s: $N=%d, T=%d, a=%s, \tau_{\max}=%d$" %(model_name, N, T, auto) 
                            +"\n" + r"%s, $\alpha=%s, \tau_{\max}=%d$" %(name[ci_test], pc_alpha, tau_max),
         fontsize=4, ha='right', va='top')
    elif which == 'sample_size':
        fig.text(0.5, 0., r"Sample size $T$", fontsize=8,
            horizontalalignment='center', va='bottom')
        fig.text(1., 1., r"%s: $N=%d, a=%s$" %(model_name, N, auto) 
                            +"\n" + r"%s, $\alpha=%s, \tau_{\max}=%d$" %(name[ci_test], pc_alpha, tau_max),
         fontsize=6, ha='right', va='top')
        fig2.text(0.5, 0., r"Sample size $T$", fontsize=5,
            horizontalalignment='center', va='bottom')

        fig2.text(1., 1., r"%s: $N=%d, a=%s$, %s" %(model_name, N, auto, name[ci_test]) 
                            +"\n" + r"$\alpha=%s, \tau_{\max}=%d$" %(pc_alpha, tau_max),
         fontsize=4, ha='right', va='top')
    elif which == 'pc_alpha':
        fig.text(0.5, 0., r"$\alpha$", fontsize=8,
            horizontalalignment='center', va='bottom')

        fig.text(1., 1., r"%s: $N=%d, T=%d, a=%s$" %(model_name, N, T, auto) 
                            +"\n" + r"%s, $\tau_{\max}=%d$" %(name[ci_test], tau_max),
         fontsize=6, ha='right', va='top')
        fig2.text(0.5, 0., r"$\alpha$", fontsize=5,
            horizontalalignment='center', va='bottom')

        fig2.text(1., 1., r"%s: $N=%d, T=%d, a=%s$, %s" %(model_name, N, T, auto,name[ci_test]) 
                            +"\n" + r"$\tau_{\max}=%d$" %(tau_max),
         fontsize=4, ha='right', va='top')
        figroc.text(1., 1., r"%s: $N=%d, T=%d, a=%s$" %(model_name, N, T, auto) 
                            +"\n" + r"%s, $\tau_{\max}=%d$" %(name[ci_test], tau_max),
         fontsize=6, ha='right', va='top')


    if which == "pc_alpha":
        for axname in axes:
            ax = axes[axname]
            ax.set_xticklabels([str(p) for p in paras], rotation=75, horizontalalignment="center",fontsize = 4)

    fig.subplots_adjust(left=0.07, right=0.93, hspace=.3, bottom=0.12, top=0.85, wspace=.3)
    fig2.subplots_adjust(left=0.2, right=0.8, hspace=.3, bottom=0.15, top=0.8, wspace=0.05)
    if which == 'pc_alpha':
        figroc.subplots_adjust(left=0.1, right=0.9, hspace=.3, bottom=0.12, top=0.85, wspace=0.25)
        print("Saving P-R curve plot to %s" %(save_folder + '%s_pr_curve.%s' %(save_suffix, save_type)))
        figroc.savefig(save_folder + '%s_pr_curve.%s' %(save_suffix, save_type))
        figalpha.subplots_adjust(left=0.2, right=0.8, hspace=.3, bottom=0.15, top=0.8, wspace=0.05)
    print("Saving plot to %s" %(save_folder + '%s.%s' %(save_suffix, save_type)))
    fig.savefig(save_folder + '%s.%s' %(save_suffix, save_type))
    print("Saving FPR plot to %s" %(save_folder + '%s_fpr.%s' %(save_suffix, save_type)))
    fig2.savefig(save_folder + '%s_fpr.%s' %(save_suffix, save_type))
    plot_files.append(save_folder + '%s.%s' %(save_suffix, save_type))
    plot_files_fpr.append(save_folder + '%s_fpr.%s' %(save_suffix, save_type))

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
    if "pcmci+" in method:

        if "bootstrap" in method:
            color_scaling = [adjust_lightness('green', amount=i) for i in [3.4,2.4,1,0]]
            return color_scaling[2]
        else: return 'orange'
    if "lpcmci" in method:
        if "bootstrap" in method:
            color_scaling = [adjust_lightness('green', amount=i) for i in [3.4,2.4,1,0]]
            return color_scaling[-2]
        else: return 'magenta'
        
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
    elif 'lpcmci' in method:
        return r"LPCMCI"
    else:
        return method

def links_from_N(num_nodes):
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
    plot_files = []
    plot_files_fpr = []
    paper = True
    SM = False
    run_anova = False

    fpr_precision = 'precision'

    methods = []

    methods += [
        'standard_pcmci+',
        'bootstrap_pcmci+',
        #'pcalg', #for fig 15/16
        #'bootstrap_pcalg' #for fig 15/16
        #'lpcmci' #for fig17/18
        #'bootstrap_lpcmci' #for fig17/18
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
        T_here = [500]
        N_here = [5]
        num_rows = 4
        tau_max = 5
        vary_auto = [0., 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 0.95, 0.98, 0.99, 0.999]
        pc_alpha_here = [0.01]
        min_coeff = 0.1
        coeff = 0.5       
        frac_unobserved = 0.
        contemp_fraction = 0.3
        max_true_lag = 5
        n_bs= [25,50,100,200]

        for T in T_here: 
         for N in N_here:
          n_links = links_from_N(N)  
          for pc_alpha in pc_alpha_here:
            para_setup_name = (variant, N, n_links, min_coeff, coeff, contemp_fraction, frac_unobserved,  
                            max_true_lag, T, ci_test, pc_alpha, tau_max,n_bs) 

            save_suffix = '%s-'*len(para_setup_name) % para_setup_name
            save_suffix = save_suffix[:-1]

            print(save_suffix)
            draw_it(paras=vary_auto, which='auto')  

    elif 'highdim' in variant:
        T_here = [500]
        vary_N =   [2,3,5,10,20,30,40]
        auto_here = [0.95] 
        num_rows = 4
        contemp_fraction = 0.3
        frac_unobserved = 0.
        max_true_lag = 5
        tau_max = 5
        min_coeff = 0.1
        coeff = 0.5
        pc_alpha_here = [0.01]
        n_bs = [25,50,100,200]
        
        for T in T_here:
            for auto in auto_here:
                  for pc_alpha in pc_alpha_here:
                    para_setup_name = (variant, min_coeff, coeff, auto, contemp_fraction, frac_unobserved, max_true_lag, T, 
                                            ci_test, pc_alpha, tau_max, n_bs)
                    save_suffix = '%s-'*len(para_setup_name) % para_setup_name
                    save_suffix = save_suffix[:-1]

                    draw_it(paras=vary_N, which='N')  

    elif  'sample_size' in variant:
        vary_T = [100,200,500,1000]
        N_here = [5]
        auto_here = [0.95]
        num_rows = 4
        min_coeff = 0.1
        coeff = 0.5        
        pc_alpha_here = [0.01]
        contemp_fraction = 0.3
        frac_unobserved = 0.
        max_true_lag = 5
        tau_max = 5

        n_bs = [25,50,100,200]
        for N in N_here:
          n_links = links_from_N(N)   
          for auto in auto_here:
                for pc_alpha in pc_alpha_here:
                    para_setup_name = (variant, N, n_links, min_coeff, coeff, contemp_fraction, frac_unobserved, 
                                        max_true_lag, auto, ci_test, pc_alpha, tau_max, n_bs)
                    save_suffix = '%s-'*len(para_setup_name) % para_setup_name
                    save_suffix = save_suffix[:-1]
                    print(save_suffix)
                    draw_it(paras=vary_T, which='sample_size')  

    if 'tau_max' in variant:
        T_here = [500]
        N_here = [5]
        auto_here = [0.95] 
        num_rows = 4
        vary_tau_max = [5, 10, 15, 20, 25, 30, 35, 40]
        min_coeff = 0.1
        coeff = 0.5
        pc_alpha_here = [0.01]
        contemp_fraction = 0.3
        frac_unobserved = 0.
        max_true_lag = 5
        n_bs = [25,50,100,200]

        for T in T_here:
         for N in N_here:
          n_links = links_from_N(N)   
          for auto in auto_here:
                for pc_alpha in pc_alpha_here:
                    para_setup_name = (variant, N, n_links, min_coeff, coeff, contemp_fraction, frac_unobserved,
                                      max_true_lag, auto, T, ci_test, pc_alpha, n_bs)
                    save_suffix = '%s-'*len(para_setup_name) % para_setup_name
                    save_suffix = save_suffix[:-1]

                    print(save_suffix)
                    draw_it(paras=vary_tau_max, which='tau_max')  

    if 'pc_alpha' in variant:
        T_here = [200]
        N_here = [10]
        auto_here = [0.95] 
        num_rows = 4
        tau_max = 5
        # Here set for fig3, change to: vary_pc_alpha = [0.001,0.005,0.01,0.02,0.05] for fig 4
        vary_pc_alpha = [0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01,0.02,0.05,0.1,0.2,0.3,
                        0.4, 0.5, 0.6, 0.8, 0.9, 0.95, 0.98, 0.99, 0.999]
        min_coeff = 0.1
        coeff = 0.5
        aggregation= "majority" #""alternative
        contemp_fraction = 0.3
        if "lpcmci" in variant: frac_unobserved= 0.3
        else: frac_unobserved = 0.
        max_true_lag = 5
        n_bs = [25,50,100,200]
        if "nonlinear" in variant: n_bs=[400]

        for T in T_here:
         for N in N_here:
          n_links = links_from_N(N)   
          for auto in auto_here:
            para_setup_name = (variant, N, n_links, min_coeff, coeff, contemp_fraction, frac_unobserved,
                                max_true_lag, auto, T, ci_test, tau_max, n_bs)
            save_suffix = '%s-'*len(para_setup_name) % para_setup_name
            save_suffix = save_suffix[:-1]
            print(save_suffix)
            draw_it(paras=vary_pc_alpha, which='pc_alpha')  