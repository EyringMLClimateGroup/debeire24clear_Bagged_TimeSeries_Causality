import os, time, sys, psutil
import numpy as np
import scipy, math
import tigramite
import tigramite.data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
import generate_data_mod as mod
import statsmodels
import statsmodels.tsa.api as tsa
import metrics_mod
import scipy.stats
import mpi
import pickle 
from copy import deepcopy
from matplotlib import pyplot
import socket 

try:
    arg = sys.argv
    num_cpus = int(arg[1])
    samples = int(arg[2])
    verbosity = int(arg[3])
    config_list = list(arg)[4:]
    num_configs = len(config_list)
except:
    arg = ''
    num_cpus = 2
    samples = 100
    verbosity = 2
    config_list = ["toymodel-9-9-0.4-0.4-0.98-0.3-2-500-par_corr-pcmcifast-majority-same-0.05"]
    
num_configs = len(config_list)

time_start = time.time()

if verbosity > 2:
    plot_data = True
else:
    plot_data = False


def calculate(para_setup):


    para_setup_string, sam = para_setup

    ## Strange behaviour on cluster...
    paras = para_setup_string.split('-')
    paras = [w.replace("'","") for w in paras]
    # print paras

    model = str(paras[0])
    N = int(paras[1])
    n_links = int(paras[2])
    min_coeff = float(paras[3])
    coeff = float(paras[4])
    auto = float(paras[5])
    contemp_fraction = float(paras[6])
    frac_unobserved = float(paras[7])
    max_true_lag = int(paras[8])
    T = int(paras[9])

    ci_test = str(paras[10])
    method = str(paras[11])    
    pc_alpha = str(paras[12])
    tau_max = int(paras[13])
    n_bs = int(paras[14])
    N_draw = int(paras[15])
    #############################################
    ##  Data
    #############################################

    addnoise = False
    addtrend = False
        
    def lin_f(x): return x
    def f2(x): return (x + 5. * x**2 * np.exp(-x**2 / 20.))

    if model == 'example1':
        model_seed = None   #int(model.split('_')[1])
               
        links ={0: [((0, -1), auto, lin_f),
                    ((1, -1), coeff, lin_f)
                    ],
                1: [((1, -1), auto, lin_f), 
                    ],                                 
                }
        noises = [np.random.randn for j in range(len(links))]
        observed_vars = range(len(links))
    
    elif model == 'autobidirected':
        if verbosity > 999:
            model_seed = verbosity - 1000
        else:
            model_seed = sam

        random_state = np.random.RandomState(model_seed)

        links ={
                0: [((0, -1), auto, lin_f), ((1, -1), coeff, lin_f)],
                1: [],
                2: [((2, -1), auto, lin_f), ((1, -1), coeff, lin_f)],                                
                3: [((3, -1), auto, lin_f), ((2, -1), min_coeff, lin_f)],                                
                }
        observed_vars = [0, 2, 3]

        noises = [random_state.randn for j in range(len(links))]

        data_all, nonstationary = mod.generate_nonlinear_contemp_timeseries(
            links=links, T=T, noises=noises, random_state=random_state)
        data = data_all[:,observed_vars]

    elif 'random' in model:
        if '_lineargaussian' in model:

            coupling_funcs = [lin_f]

            noise_types = ['gaussian'] #, 'weibull', 'uniform']
            noise_sigma = (0.5, 2)

        elif '_linearmixed' in model:

            coupling_funcs = [lin_f]

            noise_types = ['gaussian', 'weibull']
            noise_sigma = (0.5, 2)

        elif '_nonlinearmixed' in model:

            coupling_funcs = [lin_f, f2]

            noise_types = ['gaussian', 'gaussian', 'weibull']
            noise_sigma = (0.5, 2)

        elif '_nonlineargaussian' in model:

            coupling_funcs = [lin_f, f2]

            noise_types = ['gaussian']
            noise_sigma = (0.5, 2)

        if coeff < min_coeff:
            min_coeff = coeff
        couplings = list(np.arange(min_coeff, coeff+0.1, 0.1))
        couplings += [-c for c in couplings]

        auto_deps = list(np.arange(max(0., auto-0.3), auto+0.01, 0.05))

        # Models may be non-stationary. Hence, we iterate over a number of seeds
        # to find a stationary one regarding network topology, noises, etc
        if verbosity > 999:
            model_seed = verbosity - 1000
        else:
            model_seed = sam
        
        for ir in range(1000):
            # np.random.seed(model_seed)
            random_state = np.random.RandomState(model_seed)

            N_all = math.floor((N/(1.-frac_unobserved)))
            n_links_all = math.ceil(n_links/N * N_all)
            observed_vars = np.sort(random_state.choice(range(N_all), 
                size=math.ceil((1.-frac_unobserved)*N_all), replace=False)).tolist()
            
            links = mod.generate_random_contemp_model(
                N=N_all, L=n_links_all,   
                coupling_coeffs=couplings,   
                coupling_funcs=coupling_funcs,   
                auto_coeffs=auto_deps,   
                tau_max=max_true_lag,   
                contemp_fraction=contemp_fraction,  
                # num_trials=1000,  
                random_state=random_state)

            class noise_model:
                def __init__(self, sigma=1):
                    self.sigma = sigma
                def gaussian(self, T):
                    # Get zero-mean unit variance gaussian distribution
                    return self.sigma*random_state.randn(T)
                def weibull(self, T): 
                    # Get zero-mean sigma variance weibull distribution
                    a = 2
                    mean = scipy.special.gamma(1./a + 1)
                    variance = scipy.special.gamma(2./a + 1) - scipy.special.gamma(1./a + 1)**2
                    return self.sigma*(random_state.weibull(a=a, size=T) - mean)/np.sqrt(variance)
                def uniform(self, T): 
                    # Get zero-mean sigma variance uniform distribution
                    mean = 0.5
                    variance = 1./12.
                    return self.sigma*(random_state.uniform(size=T) - mean)/np.sqrt(variance)

            noises = []
            for j in links:
                noise_type = random_state.choice(noise_types)
                sigma = noise_sigma[0] + (noise_sigma[1]-noise_sigma[0])*random_state.rand()
                noises.append(getattr(noise_model(sigma), noise_type))

            data_all_check, nonstationary = mod.generate_nonlinear_contemp_timeseries(
                links=links, T= T+10000, noises=noises, random_state=random_state)

            # If the model is stationary, break the loop
            if not nonstationary:
                data_all = data_all_check[:T]
                data = data_all[:,observed_vars]
                break
            else:
                print("Trial %d: Not a stationary model" % ir)
                model_seed += 10000
    else:
        raise ValueError("model %s not known"%model)

    if nonstationary:
        raise ValueError("No stationary model found: %s" % model)
        # print("Nonstationary: %s" % model)

    true_graph = np.zeros((N, N, tau_max + 1), dtype = '<U3')
    true_graph[:] = ""
    for v in range(N): 
        for parent in links[v]:
            ## parent = ((0, -1), .8, 'linear')
            u = parent[0][0]
            lag = parent[0][1]
            coeff = parent[1]
            # Ignore type of functional dependency
            coupling = parent[2]
            # Consider only cross-links
            # if u != v:
                # Get TPR of this link
            if coeff != 0.:
                true_graph[u,v,abs(lag)] = "-->"
                if lag == 0:
                    true_graph[v,u,abs(lag)] = "<--"

    if verbosity > 0:
        print("True Links")
        for j in links:
            print (j, links[j])
        print("observed_vars = ", observed_vars)
        print("True PAG")
        if tau_max > 0:
            for lag in range(tau_max+1):
                print(true_graph[:,:,lag])
        else:
            print(true_graph.squeeze())

    if addnoise:
        data += noise_lev*np.random.RandomState(sam).randn(*data.shape)

    if addtrend:
        data += trend_lev*np.sin(2.*np.pi/(T/6.) * np.arange(T).reshape(T, 1))



    if plot_data:
        print("PLOTTING")
        for j in range(N):
            pyplot.plot(data[:, j])

        pyplot.show()


    computation_time_start = time.time()

    dataframe = pp.DataFrame(data)

    #############################################
    ##  Methods
    #############################################

    if pc_alpha == 'none':
        pc_alpha = None
    else:
        pc_alpha = float(pc_alpha)

    # Specify conditional independence test object
    if ci_test == 'par_corr':
        cond_ind_test = ParCorr(
            significance='analytic', 
            recycle_residuals=False)
    elif ci_test == 'cmi_knn':
        cond_ind_test = CMIknn(knn=0.1, 
            sig_samples=500,
            sig_blocklength=1)
    elif ci_test == 'gp_dc':             
        cond_ind_test = GPDC(
            recycle_residuals=False)
    elif ci_test == 'oracle':             
        cond_ind_test = OracleCI(link_coeffs=links, 
            observed_vars=observed_vars)

    if method == 'ground_truth':
        graph = true_graph

    elif 'standard_pcmci+' in method:
        reset_lagged_links = False
        if 'resetlagged' in method: reset_lagged_links = True

        max_conds_px = None
        if 'allpx0' in method: max_conds_px = 0

        max_conds_px_lagged = None
        if 'laggedpx0' in method: max_conds_px_lagged = 0

        val_matrix_results = np.empty((N_draw,N,N,tau_max+1))
        graph_results = np.empty((N_draw,N,N,tau_max+1),dtype='<U3')

        #Generate N_draw stationary ime series with same SCM (but new noises) and re-estimate links with standard PCMCI+
        start_retry = 0
        for draw_num in range(N_draw):
            for retry in range(start_retry+draw_num,start_retry+draw_num+1000):
                random_state_draw = np.random.RandomState(model_seed+retry)
                data_all_check, nonstationary = mod.generate_nonlinear_contemp_timeseries(
                                                links=links, T= T+10000, noises=noises, random_state=random_state_draw)
                if not nonstationary:
                    break
                else:
                    start_retry += 1
            if nonstationary:
                raise ValueError("This model is not stationary!")

            data_all = data_all_check[:T]
            data_draw = data_all[:,observed_vars]
            dataframe_draw = pp.DataFrame(data_draw)
            pcmci = PCMCI(
                dataframe=dataframe_draw, 
                cond_ind_test=cond_ind_test,
                verbosity=verbosity)

            pcmcires = pcmci.run_pcmciplus(
                tau_min=0,
                tau_max=tau_max,
                pc_alpha=pc_alpha,
                contemp_collider_rule='majority',
                conflict_resolution=True,
                reset_lagged_links=reset_lagged_links,
                max_conds_dim=None,
                max_conds_py=None,
                max_conds_px=max_conds_px,
                max_conds_px_lagged=max_conds_px_lagged,
                fdr_method='none',)
            graph = pcmcires['graph']
            val_min = np.abs(pcmcires['val_matrix'])
            val_matrix_results[draw_num,...]= val_min
            graph_results[draw_num,...]= graph
            graph_bool= graph
            max_cardinality = np.ones(graph_bool.shape, dtype='int')
        results= {}
        results["val_matrix"] = val_matrix_results
        results["graph"] = graph_results
        redraw_pcmci_res = pcmci.return_summary_results(results)
        graph = redraw_pcmci_res['most_frequent_links']
        val_min = np.abs(redraw_pcmci_res['val_matrix_mean'])
        link_freq = redraw_pcmci_res['link_frequency']
        #print(link_freq)
    elif method == "bootstrap_pcmci+":
        reset_lagged_links = False
        if 'resetlagged' in method: reset_lagged_links = True

        max_conds_px = None
        if 'allpx0' in method: max_conds_px = 0

        max_conds_px_lagged = None
        if 'laggedpx0' in method: max_conds_px_lagged = 0
    
        link_freq_save = np.empty((N_draw,N,N,tau_max+1))

        #Generate N_draw_bs stationary time series with same SCM (but new noises) and re-estimate links with standard PCMCI+
        start_retry = 0
        for draw_num in range(N_draw):
            for retry in range(start_retry+draw_num,start_retry+draw_num+1000):
                random_state_draw = np.random.RandomState(model_seed+retry)
                data_all_check, nonstationary = mod.generate_nonlinear_contemp_timeseries(
                                                links=links, T= T+10000, noises=noises, random_state=random_state_draw)
                if not nonstationary:
                    break
                else:
                    start_retry += 1
            if nonstationary:
                raise ValueError("This model is not stationary!")

            data_all = data_all_check[:T]
            data_draw = data_all[:,observed_vars]
            dataframe_draw = pp.DataFrame(data_draw)
            pcmci = PCMCI(
                dataframe=dataframe_draw, 
                cond_ind_test=cond_ind_test,
                verbosity=verbosity)

            pcmci_arg= {
                "tau_min": 0,
                "tau_max": tau_max,
                "pc_alpha": pc_alpha,
                "contemp_collider_rule": 'majority',
                "conflict_resolution": True,
                "reset_lagged_links": reset_lagged_links,
                "max_conds_dim": None,
                "max_conds_py": None,
                "max_conds_px": max_conds_px,
                "max_conds_px_lagged": max_conds_px_lagged,
                "fdr_method":'none'}
            pcmcires = pcmci.run_bootstrap_of('run_pcmciplus',pcmci_arg,boot_samples =n_bs, boot_blocklength=1, seed=model_seed+draw_num )
            graph = pcmcires['summary_results']['most_frequent_links']
            link_freq = pcmcires['summary_results']['link_frequency']
            val_min = np.abs(pcmcires['summary_results']['val_matrix_mean'])
            link_freq_save[draw_num] = link_freq
        link_freq= link_freq_save.mean(axis=0)
        max_cardinality = np.ones(graph.shape, dtype='int')

    else:
        raise ValueError("%s not implemented." % method)


    computation_time_end = time.time()
    computation_time = computation_time_end - computation_time_start

    if verbosity > 1 and sam > 9:
        pcmci2 = pcmci.run_pcmci(
            tau_min=1,
            tau_max=tau_max,
            pc_alpha=pc_alpha,
            max_conds_dim=None,
            max_conds_py=None,
            max_conds_px=None,
            fdr_method='none',)

        graph2 = pcmci.convert_to_string_graph(pcmci2['graph'])
        if np.any((true_graph[:,:,1:]=="")*(graph[:,:,1:]!="")*(graph2[:,:,1:]=="")): # \
            print("True Links")
            for j in links:
                print (j, links[j])
            print("observed_vars = ", observed_vars)
            print("True PAG")
            if tau_max > 0:
                for lag in range(tau_max+1):
                    print("Lag ", lag)
                    print(true_graph[:,:,lag])
                    print(graph[:,:,lag])
            else:
                print(true_graph.squeeze())
                print(graph.squeeze())
            raise ValueError("Look here ", sam)


    if ci_test == 'oracle':
        if not np.all(true_graph==graph):
            print("True Links")
            for j in links:
                print (j, links[j])
            print("observed_vars = ", observed_vars)
            print("True vs Estimated PAG")
            for lag in range(tau_max+1):
                print("True at lag = ", lag)
                print(true_graph[:,:,lag])
                print("Est")
                print(graph[:,:,lag])
            raise ValueError("Wrong graph in Oracle case for ", para_setup_string, model_seed)

    return {
            'true_graph':true_graph,
            'val_min':val_min,
            'max_cardinality':max_cardinality,

            # Method results
            'computation_time': computation_time,
            'graph': graph,
            'link_frequency': link_freq 
            }


def process_chunks(job_id, chunk):


    results = {}
    num_here = len(chunk)
    time_start_process = time.time()
    for isam, config_sam in enumerate(chunk):
        print(config_sam)
        results[config_sam] = calculate(config_sam)

        current_runtime = (time.time() - time_start_process)/3600.
        current_runtime_hr = int(current_runtime)
        current_runtime_min = 60.*(current_runtime % 1.)
        estimated_runtime = current_runtime * num_here / (isam+1.)
        estimated_runtime_hr = int(estimated_runtime)
        estimated_runtime_min = 60.*(estimated_runtime % 1.)
        print("job_id %d index %d/%d: %dh %.1fmin / %dh %.1fmin:  %s" % (
            job_id, isam+1, num_here, current_runtime_hr, current_runtime_min, 
                                    estimated_runtime_hr, estimated_runtime_min,  config_sam))
    return results


def master():

    print("Starting with num_cpus = ", num_cpus)

    all_configs = dict([(conf, {'results':{}, 
        "graphs":{}, 
        "val_min":{}, 
        "max_cardinality":{}, 
        "link_frequency":{},
        "true_graph":{}, 
        "computation_time":{},} ) for conf in config_list])
    print(config_list)
    job_list = [(conf, i) for i in range(samples) for conf in config_list]

    num_tasks = len(job_list)

    num_jobs = min(num_cpus-1, num_tasks)

    def split(a, n):
        k, m = len(a) // n, len(a) % n
        return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


    config_chunks = split(job_list, num_jobs)

    print("num_tasks %s" % num_tasks)
    print("num_jobs %s" % num_jobs)
    ## Send
    for job_id, chunk in enumerate(config_chunks):
        # print chunk
        # sys.exit(0)

        print("submit %d / %d" % (job_id, len(config_chunks)))
        mpi.submit_call("process_chunks", (job_id, chunk), id = job_id)
    mpi.info()
    ## Retrieve  
    for job_id, chunk in enumerate(config_chunks):
        print("\nreceive %s" % job_id)
        tmp = mpi.get_result(id=job_id)
        for conf_sam in list(tmp.keys()):
            config = conf_sam[0]
            sample = conf_sam[1]
            all_configs[config]['results'][sample] = tmp[conf_sam]


    print("\nsaving all configs...")

    for conf in list(all_configs.keys()):

        all_configs[conf]['graphs'] = np.zeros((samples, ) + all_configs[conf]['results'][0]['graph'].shape, dtype='<U3')

        all_configs[conf]['true_graphs'] = np.zeros((samples, ) + all_configs[conf]['results'][0]['true_graph'].shape, dtype='<U3')
        all_configs[conf]['val_min'] = np.zeros((samples, ) + all_configs[conf]['results'][0]['val_min'].shape)
        all_configs[conf]['max_cardinality'] = np.zeros((samples, ) + all_configs[conf]['results'][0]['max_cardinality'].shape)
        all_configs[conf]['link_frequency'] = np.zeros((samples, ) + all_configs[conf]['results'][0]['link_frequency'].shape)
        all_configs[conf]['computation_time'] = [] 


        for i in list(all_configs[conf]['results'].keys()):
            all_configs[conf]['graphs'][i] = all_configs[conf]['results'][i]['graph']
            all_configs[conf]['true_graphs'][i] = all_configs[conf]['results'][i]['true_graph']
            all_configs[conf]['val_min'][i] = all_configs[conf]['results'][i]['val_min']
            all_configs[conf]['max_cardinality'][i] = all_configs[conf]['results'][i]['max_cardinality']
            all_configs[conf]['link_frequency'][i] = all_configs[conf]['results'][i]['link_frequency']
            all_configs[conf]['computation_time'].append(all_configs[conf]['results'][i]['computation_time'])
        
        del all_configs[conf]['results']

        file_name = './%s' %(conf) #PATH TO SAVED DATA, ADJUST IF NEEDED
   
        print("dump ", file_name.replace("'", "").replace('"', '') + '.dat')
        file = open(file_name.replace("'", "").replace('"', '') + '.dat', 'wb')
        pickle.dump(all_configs[conf], file, protocol=-1)        
        file.close()

        # Directly compute metrics and save in much smaller dict
        para_setup_str = tuple(conf.split("-"))
        metrics = metrics_mod.get_counts(para_setup_str, from_file = False)
        if metrics is not None:
            for metric in metrics:
                if metric != 'computation_time':
                    print(f"{metric:30s} {metrics[metric][0]: 1.2f} +/-{metrics[metric][1]: 1.2f} ")
                else:
                    # print(metrics[metric])
                    print(f"{metric:30s} {metrics[metric][0]: 1.2f} +/-[{metrics[metric][1][0]: 1.2f}, {metrics[metric][1][1]: 1.2f}]")

            print("Metrics dump ", file_name.replace("'", "").replace('"', '') + '_metrics.dat')
            file = open(file_name.replace("'", "").replace('"', '') + '_metrics.dat', 'wb')
            pickle.dump(metrics, file, protocol=-1)        
            file.close()       


    time_end = time.time()
    print('Run time in hours ', (time_end - time_start)/3600.)
    

mpi.run(verbose=False)




