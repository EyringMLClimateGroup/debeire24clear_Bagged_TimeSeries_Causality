import sys, os
import numpy as np
import pickle, pickle
import scipy.stats
from scipy.optimize import leastsq
import scipy
import statsmodels.api as sm # recommended import according to the docs
from copy import deepcopy

#Script to compute the mean absolute frequency errors of Fig5B 
folder_name = './' #PATH OF NUMERICAL EXP
save_folder= os.path.join(folder_name,'')


def get_results(para_setup):

    name_string = '%s-'*len(para_setup)  # % para_setup
    name_string = name_string[:-1]
    file_name = folder_name + name_string % tuple(para_setup)

    try:
        print(file_name)
        print(file_name.replace("'", "").replace('"', ''))
        print("load  ", file_name.replace("'", "").replace('"', '') + '.dat')
        results = pickle.load(open(file_name.replace("'", "").replace('"', '') + '.dat', 'rb'), encoding='latin1')
    except Exception as e:
        print('failed '  , tuple(para_setup))
        print(e)
        return None
        # raise RuntimeError("File not found")

    return results


def get_metrics_from_file(para_setup):

    name_string = '%s-'*len(para_setup)  # % para_setup
    name_string = name_string[:-1]

    try:
        print("load from metrics file  %s_metrics.dat " % (folder_name + name_string % tuple(para_setup)))
        results = pickle.load(open(folder_name + name_string % tuple(para_setup) + '_metrics.dat', 'rb'), encoding='latin1')
    except:
        print('failed from metrics file '  , tuple(para_setup))
        return None
        # raise RuntimeError("File not found")

    return results



def get_masks(true_graphs):


    n_realizations, N, N, taumaxplusone = true_graphs.shape
    print(n_realizations, N, N, taumaxplusone)
    tau_max = taumaxplusone - 1

    cross_mask = np.repeat(np.identity(N).reshape(N,N,1)==False, tau_max + 1, axis=2).astype('bool')
    cross_mask[range(N),range(N),0]=False
    contemp_cross_mask_tril = np.zeros((N,N,tau_max + 1)).astype('bool')
    contemp_cross_mask_tril[:,:,0] = np.tril(np.ones((N, N)), k=-1).astype('bool')

    lagged_mask = np.ones((N,N,tau_max + 1)).astype('bool')
    lagged_mask[:,:,0] = 0
    # auto_mask = np.ones((N,N,tau_max + 1)).astype('bool')
    auto_mask = lagged_mask*(cross_mask == False)

    any_mask = np.ones((N,N,tau_max + 1)).astype('bool')
    any_mask[:,:,0] = contemp_cross_mask_tril[:,:,0]

    # n_realizations = len(results['graphs'])
    # true_graphs = results['true_graphs']

    cross_mask = np.repeat(cross_mask.reshape(1, N,N,tau_max + 1), n_realizations, axis=0)
    contemp_cross_mask_tril = np.repeat(contemp_cross_mask_tril.reshape(1, N,N,tau_max + 1), n_realizations, axis=0)
    lagged_mask = np.repeat(lagged_mask.reshape(1, N,N,tau_max + 1), n_realizations, axis=0)
    auto_mask = np.repeat(auto_mask.reshape(1, N,N,tau_max + 1), n_realizations, axis=0)
    any_mask = np.repeat(any_mask.reshape(1, N,N,tau_max + 1), n_realizations, axis=0)
    return cross_mask, contemp_cross_mask_tril, lagged_mask, auto_mask, any_mask, tau_max

def _get_match_score(true_link, pred_link):
    if true_link == "" or pred_link == "": return 0
    count = 0
    # If left edgemark is correct add 1
    if true_link[0] == pred_link[0]:
        count += 1
    # If right edgemark is correct add 1
    if true_link[2] == pred_link[2]:
        count += 1
    return count

def _get_abs_link_diff(true_link_freq, pred_link_freq):
    #print(np.abs(true_link_freq-pred_link_freq))
    return np.abs(true_link_freq-pred_link_freq)

match_func = np.vectorize(_get_match_score, otypes=[int])
afd_func = np.vectorize(_get_abs_link_diff, otypes=[float])

def _get_conflicts(pred_link):
    if pred_link == "": return 0
    count = 0
    # If left edgemark is conflict add 1
    if pred_link[0] == 'x':
        count += 1
    # If right edgemark is conflict add 1
    if pred_link[2] == 'x':
        count += 1
    return count
conflict_func = np.vectorize(_get_conflicts, otypes=[int]) 

def _get_unoriented(true_link):
    if true_link == "": return 0
    count = 0
    # If left edgemark is unoriented add 1
    if true_link[0] == 'o':
        count += 1
    # If right edgemark is unoriented add 1
    if true_link[2] == 'o':
        count += 1
    return count
unoriented_func = np.vectorize(_get_unoriented, otypes=[int]) 

def get_numbers(metrics, orig_true_graphs, orig_pred_graphs, true_link_freq, pred_link_freq, boot_samples=500):

    #some pre-defined masks
    cross_mask, contemp_cross_mask_tril, lagged_mask, auto_mask, any_mask, tau_max = get_masks(orig_true_graphs)
    #Mask values where the true_freq and pred_freq are 1
    true_freq_not_1_mask = (~any_mask)*np.where(true_link_freq<1.,False,True)*np.where(pred_link_freq<1.,False,True)

    n_realizations = len(orig_pred_graphs)
    metrics_dict = {}

    pred_graphs = orig_pred_graphs
    true_graphs = orig_true_graphs

    # Adjacency Absolute Link Frequency Difference, separated by lagged/auto/contemp
    #Lagged links (and absent, existing, all)
    true_link_masked = np.ma.array(orig_true_graphs,mask= ~(cross_mask*lagged_mask))
    true_link_freq_masked = np.ma.array(true_link_freq,mask= ~(cross_mask*lagged_mask))
    pred_link_freq_masked = np.ma.array(pred_link_freq,mask= ~(cross_mask*lagged_mask))
    metrics_dict['adj_lagged_abs_freq_diff'] = ((afd_func(true_link_freq_masked,pred_link_freq_masked)).mean(axis=(1,2,3)),1)
    metrics_dict['adj_lagged_abs_freq_diff_existing'] = ((afd_func(np.ma.where(true_link_masked!="",true_link_freq_masked,np.ma.array([0],mask=True)),np.ma.where(true_link_masked!="",pred_link_freq_masked,np.ma.array([0],mask=True)))).mean(axis=(1,2,3)),1)
    metrics_dict['adj_lagged_abs_freq_diff_absent'] = ((afd_func(np.ma.where(true_link_masked=="",true_link_freq_masked,np.ma.array([0],mask=True)),np.ma.where(true_link_masked=="",pred_link_freq_masked,np.ma.array([0],mask=True)))).mean(axis=(1,2,3)),1)
    
    #Auto links (and absent, existing, all)
    true_link_masked = np.ma.array(orig_true_graphs,mask= ~auto_mask)
    true_link_freq_masked = np.ma.array(true_link_freq,mask= ~auto_mask)
    pred_link_freq_masked = np.ma.array(pred_link_freq,mask= ~auto_mask)
    metrics_dict['adj_auto_abs_freq_diff'] = ((afd_func(true_link_freq_masked,pred_link_freq_masked)).mean(axis=(1,2,3)),1)
    metrics_dict['adj_auto_abs_freq_diff_existing'] = ((afd_func(np.ma.where(true_link_masked!="",true_link_freq_masked,np.ma.array([0],mask=True)),np.ma.where(true_link_masked!="",pred_link_freq_masked,np.ma.array([0],mask=True)))).mean(axis=(1,2,3)),1)
    metrics_dict['adj_auto_abs_freq_diff_absent'] = ((afd_func(np.ma.where(true_link_masked=="",true_link_freq_masked,np.ma.array([0],mask=True)),np.ma.where(true_link_masked=="",pred_link_freq_masked,np.ma.array([0],mask=True)))).mean(axis=(1,2,3)),1)

    #Contemp links (absent existing and all)
    true_link_masked = np.ma.array(orig_true_graphs,mask= ~contemp_cross_mask_tril)
    true_link_freq_masked = np.ma.array(true_link_freq,mask= ~contemp_cross_mask_tril)
    pred_link_freq_masked = np.ma.array(pred_link_freq,mask= ~contemp_cross_mask_tril)
    metrics_dict['adj_contemp_abs_freq_diff'] = ((afd_func(true_link_freq_masked,pred_link_freq_masked)).mean(axis=(1,2,3)),1)
    metrics_dict['adj_contemp_abs_freq_diff_existing'] = ((afd_func(np.ma.where(true_link_masked!="",true_link_freq_masked,np.ma.array([0],mask=True)),np.ma.where(true_link_masked!="",pred_link_freq_masked,np.ma.array([0],mask=True)))).mean(axis=(1,2,3)),1)
    metrics_dict['adj_contemp_abs_freq_diff_absent'] = ((afd_func(np.ma.where(true_link_masked=="",true_link_freq_masked,np.ma.array([0],mask=True)),np.ma.where(true_link_masked=="",pred_link_freq_masked,np.ma.array([0],mask=True)))).mean(axis=(1,2,3)),1)

    
    #Links with different freq only (absent existing all)
    true_link_masked = np.ma.array(orig_true_graphs,mask= true_freq_not_1_mask)
    true_link_freq_masked = np.ma.array(true_link_freq,mask= true_freq_not_1_mask)
    pred_link_freq_masked = np.ma.array(pred_link_freq,mask= true_freq_not_1_mask)
    metrics_dict['adj_not1_abs_freq_diff'] = ((afd_func(true_link_freq_masked,pred_link_freq_masked)).mean(axis=(1,2,3)),1)
    metrics_dict['adj_not1_abs_freq_diff_existing'] = ((afd_func(np.ma.where(true_link_masked!="",true_link_freq_masked,np.ma.array([0],mask=True)),np.ma.where(true_link_masked!="",pred_link_freq_masked,np.ma.array([0],mask=True)))).mean(axis=(1,2,3)),1)
    metrics_dict['adj_not1_abs_freq_diff_absent'] = ((afd_func(np.ma.where(true_link_masked=="",true_link_freq_masked,np.ma.array([0],mask=True)),np.ma.where(true_link_masked=="",pred_link_freq_masked,np.ma.array([0],mask=True)))).mean(axis=(1,2,3)),1)
    
    #All links (and absent, existing, all)
    true_link_masked = np.ma.array(orig_true_graphs,mask= ~any_mask)
    true_link_freq_masked = np.ma.array(true_link_freq,mask= ~any_mask)
    pred_link_freq_masked = np.ma.array(pred_link_freq,mask= ~any_mask)
    metrics_dict['adj_anylink_abs_freq_diff'] = ((afd_func(true_link_freq_masked,pred_link_freq_masked)).mean(axis=(1,2,3)),1)      
    metrics_dict['adj_anylink_abs_freq_diff_existing'] = ((afd_func(np.ma.where(true_link_masked!="",true_link_freq_masked,np.ma.array([0],mask=True)),np.ma.where(true_link_masked!="",pred_link_freq_masked,np.ma.array([0],mask=True)))).mean(axis=(1,2,3)),1)
    metrics_dict['adj_anylink_abs_freq_diff_absent'] = ((afd_func(np.ma.where(true_link_masked=="",true_link_freq_masked,np.ma.array([0],mask=True)),np.ma.where(true_link_masked=="",pred_link_freq_masked,np.ma.array([0],mask=True)))).mean(axis=(1,2,3)),1)
    
    for metric in metrics_dict.keys():

        numerator = metrics_dict[metric][0]
        metric_boot = np.zeros(boot_samples)
        for b in range(boot_samples):
            # Store the unsampled values in b=0
            rand = np.random.randint(0, n_realizations, n_realizations)
            metric_boot[b] = numerator[rand].mean()

        metrics_dict[metric] = (100*numerator.mean(), 100*metric_boot.std())

    return metrics_dict


def get_counts(para_setup1, para_setup2, from_file=False):


    metrics = [ 'adj_' + link_type + "_" + metric_type for link_type in ['lagged', 'auto', 'contemp', 'anylink'] 
                                                       for metric_type in ['abs_freq_diff']]

    if from_file:
        metrics_from_file = get_metrics_from_file(para_setup)
        if metrics_from_file is not None:
            return metrics_from_file

    results1 = get_results(para_setup1)
    results2 = get_results(para_setup2)
    if results1 is not None and results2 is not None:

        # Same tau_max for all trials
        orig_true_graphs = results1['graphs']
        true_link_freq = results1['link_frequency']
        # Pred graphs also contain 2's for conflicting links...
        orig_pred_graphs = results2['graphs']
        pred_link_freq = results2['link_frequency']

        # print(true_graphs.shape, pred_graphs.shape, contemp_cross_mask.shape, cross_mask.shape, lagged_mask.shape, (cross_mask*lagged_mask).shape )
        metrics_dict = get_numbers(metrics, orig_true_graphs, orig_pred_graphs, true_link_freq, pred_link_freq, boot_samples=500)
        computation_time = results2['computation_time']
        metrics_dict['computation_time'] = (np.mean(np.array(computation_time)), np.percentile(np.array(computation_time), [5, 95]))
        return metrics_dict
    else:
        return None



if __name__ == '__main__':
    #Take care of using same parameters as in create_submission_fig5B.py
    save_metric= True
    N_draw=1000
    N_draw_bs = 1
    method_list = ['standard_pcmci+','bootstrap_pcmci+']
    for model in ['random_lineargaussian']: #random_lineargaussian_highdegree
        for N in [3]:
            if N == 2:
                n_links = 1
            else:
                if 'fixeddensity' in model:
                    n_links = max(N, int(0.2*N*(N-1.)/2.)) 
                elif 'highdegree' in model:
                    n_links = int(1.5*N)   
                else:
                    n_links = N
            for min_coeff in [0.1]:
                for coeff in [0.5]:
                    for auto in [0.95]:
                        for max_true_lag in [2]:
                            for contemp_fraction in [0.3]:
                                for frac_unobserved in [0.]:
                                    for T in [500]:
                                        for ci_test in ['par_corr']: 
                                            for pc_alpha in [0.01]:
                                                for tau_max in [2]:
                                                    for n_bs in [25,100,250,500,750,1000,1500,2000,2500]:
                                                        #True link frequency config
                                                        method= method_list[0]
                                                        para_setup_str1 = (model, N, n_links, min_coeff, coeff, auto, contemp_fraction, frac_unobserved, max_true_lag, T, ci_test, method, pc_alpha, tau_max,
                                                            0, N_draw)
                                                        #Bootstrap link frequency config
                                                        method= method_list[1]
                                                        para_setup_str2 = (model, N, n_links, min_coeff, coeff, auto, contemp_fraction, frac_unobserved, max_true_lag, T, ci_test, method, pc_alpha, tau_max,
                                                            n_bs, N_draw_bs)
                                                        conf2 = "-".join([str(elem) for elem in para_setup_str2])
                                                        metrics = get_counts(para_setup_str1, para_setup_str2, from_file = False)
                                                        print(conf2)
                                                        if metrics is not None:
                                                            for metric in metrics:
                                                                if metric == 'computation_time':
                                                                    print(f"{metric:30s} {metrics[metric][0]: 1.2f} +/-[{metrics[metric][1][0]: 1.2f}, {metrics[metric][1][1]: 1.2f}]")
                                                                else:
                                                                    print(f"{metric:30s} {metrics[metric][0]: 1.4f} +/-[{metrics[metric][1]:1.4f}]")
                
                                                        file_name = save_folder+ '/%s' %(conf2)
                                                        else:
                                                            pass
                                                        if save_metric:
                                                            print("Link frequency metrics dump ", file_name.replace("'", "").replace('"', '') + '_link_frequency_metrics.dat')
                                                            file = open(file_name.replace("'", "").replace('"', '') + '_link_frequency_metrics.dat', 'wb')
                                                            pickle.dump(metrics, file, protocol=-1)
                                                            file.close()