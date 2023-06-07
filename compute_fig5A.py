import tigramite
from tigramite.pcmci import PCMCI
import pickle
from tigramite.independence_tests.parcorr import ParCorr
import tigramite.data_processing as pp
from tigramite.toymodels import structural_causal_processes as toys
import tigramite.plotting as tp
import numpy as np
import scipy.stats
from matplotlib import pyplot as plt
import mpi
import os, time, sys, psutil
import generate_data_mod as mod

try:
    arg = sys.argv
    num_cores = int(arg[1])
except:
    arg = ''
    num_cores = 128  #default number of cores per CPU on our hardware


##SCM parameters
N = 3  #Number of variables
L = 3  #Number of cross links
T = 200 #Time sample size
pc_alpha = 0.01  #alpha_pc of PCMCI+
tau_min = 0
tau_max = 2
boot_samples = 1000  # number of bootstrap realizations B
scm_models = 250   #Number of models
repetions = 100   #Number of indepedendent data samples for each model

def lin_f(x): return x

class noise_model:
    def __init__(self, sigma=1,random_state = np.random.RandomState(0)):
        self.sigma = sigma
        self.random_state= random_state
    def gaussian(self, T):
        # Get zero-mean unit variance gaussian distribution
        return self.sigma*self.random_state.randn(T)
    def weibull(self, T): 
        # Get zero-mean sigma variance weibull distribution
        a = 2
        mean = scipy.special.gamma(1./a + 1)
        variance = scipy.special.gamma(2./a + 1) - scipy.special.gamma(1./a + 1)**2
        return self.sigma*(self.random_state.weibull(a=a, size=T) - mean)/np.sqrt(variance)
    def uniform(self, T): 
        # Get zero-mean sigma variance uniform distribution
        mean = 0.5
        variance = 1./12.
        return self.sigma*(self.random_state.uniform(size=T) - mean)/np.sqrt(variance)

def calculate(num_scm_model):
    #print(num_scm_model)
    verbosity = 0
    scm_seed = num_scm_model

    auto= 0.95
    coeff=0.5
    min_coeff=0.1
    coupling_funcs = [lin_f]
    noise_types = ['gaussian'] #, 'weibull', 'uniform']
    noise_sigma = (0.5, 2)

    if coeff < min_coeff:
        min_coeff = coeff
    couplings = list(np.arange(min_coeff, coeff+0.1, 0.1))
    couplings += [-c for c in couplings]
    auto_deps = list(np.arange(max(0., auto-0.3), auto+0.01, 0.05))

    #look for a stationary SCM
    for j in range(1000):
        #Create an SCM
        random_state = np.random.RandomState(scm_seed)
        links = mod.generate_random_contemp_model(
                N=N, L=L,   
                coupling_coeffs=couplings,   
                coupling_funcs=coupling_funcs,   
                auto_coeffs=auto_deps,   
                tau_max=tau_max,   
                contemp_fraction=0.3,  
                # num_trials=1000,  
                random_state=random_state)

        #Generate a time series with current SCM
        noises = []
        for k in links:
            noise_type = random_state.choice(noise_types)
            sigma = noise_sigma[0] + (noise_sigma[1]-noise_sigma[0])*random_state.rand()
            noises.append(getattr(noise_model(sigma,random_state), noise_type))

        data, nonstat = mod.generate_nonlinear_contemp_timeseries(
                links=links, T= T+10000, noises=noises, random_state=random_state)

        # If the model is stationary, we keep it
        if not nonstat:
            break
        else:
            print("Trial %d: Not a stationary model" % j)
            scm_seed += 10000

    #If no stationary model has been found
    if nonstat:
        raise ValueError("No stationary model found!")
    
    scm_graph = np.zeros((N, N, tau_max + 1), dtype = '<U3')
    scm_graph[:] = ""
    for v in range(N): 
        for parent in links[v]:
            u = parent[0][0]
            lag = parent[0][1]
            coeff = parent[1]
            # Ignore type of functional dependency
            coupling = parent[2]
            if coeff != 0.:
                scm_graph[u,v,abs(lag)] = "-->"
                if lag == 0:
                    scm_graph[v,u,abs(lag)] = "<--"

    graphs = np.empty((repetions, N, N, tau_max + 1), dtype='<U3')
    boot_graphs = np.empty((repetions, N, N, tau_max + 1), dtype='<U3')

    boot_linkfreq = np.empty((repetions, N, N, tau_max + 1))
    boot_linkfreq_mean = np.empty((N, N, tau_max + 1))
    boot_linkfreq_std = np.empty((N, N, tau_max + 1))

    true_linkfreq = np.empty((N, N, tau_max + 1))
    true_linkfreq_boot_mean= np.empty((N, N, tau_max + 1))
    true_linkfreq_boot_std= np.empty((N, N, tau_max + 1))
    
    true_graph = np.empty((N, N, tau_max + 1), dtype='<U3')
    
    #With this stationary SCM we generate "repetions" samples
    start_retry = 0
    for ir in range(repetions):
        # Generate time series from stationary SCM
        for retry in range(start_retry+ir,start_retry+ir+1000):
            random_state = np.random.RandomState(scm_seed+retry)
            data_all, nonstat = mod.generate_nonlinear_contemp_timeseries(
                links=links, T= T+10000, noises=noises, random_state=random_state)
            if not nonstat:
                break
            else:
                start_retry +=1

        if nonstat:
            raise ValueError("This model is not stationary!")
        data = data_all[:T]
        dataframe = pp.DataFrame(data)
        del data_all

        ###Run Causal discovery: PCMCI+ and Bootstrap-PCMCI+
        ##PCMCIplus to get ground truth
        pcmci = PCMCI(dataframe=dataframe,
                    cond_ind_test=ParCorr(),
                    verbosity=verbosity,
                    )
        results = pcmci.run_pcmciplus(tau_min=tau_min, tau_max=tau_max, pc_alpha=pc_alpha)
        #Save PCMCI+ graph 
        #(Frequency of links in calculated after all repetitions are finished)
        graphs[ir] = results['graph']

        ##Bootstrapped PCMCIplus
        pcmci = PCMCI(dataframe=dataframe,
            cond_ind_test=ParCorr(),
            verbosity=verbosity,
            )
        results = pcmci.run_bootstrap_of(
                method='run_pcmciplus', 
                method_args={'tau_min':tau_min, 'tau_max':tau_max, 
                'pc_alpha':pc_alpha}, 
                boot_samples=boot_samples,
                boot_blocklength=1,
                seed=ir+2565)['summary_results']
        #Save output graph and link frequency
        boot_linkfreq[ir] = results['link_frequency']
        boot_graphs[ir] = results['most_frequent_links']

    # Get PCMCI+ frequency (True/Ground truth frequency)
    summary = pcmci.return_summary_results({'graph':graphs, 'val_matrix':np.zeros(graphs.shape)})
    true_linkfreq = summary['link_frequency']
    true_graph = summary['most_frequent_links']

    # Get mean and std.dev estimate of bootstrap link frequency
    #(if we would only use one repetition, we would get a noisy estimate of the 
    boot_linkfreq_mean = boot_linkfreq.mean(axis=0)
    boot_linkfreq_std = boot_linkfreq.std(axis=0)

    #Estimate std. dev. of true link frequency with resampling
    true_freq_boot_samples = 500
    true_freq_boot = np.zeros((true_freq_boot_samples,N, N, tau_max + 1))
    for b in range(true_freq_boot_samples):
        # Store the unsampled values in b=0
        rand = np.random.randint(0, repetions, repetions)
        rand_graph = graphs[rand]
        true_freq_boot[b] = pcmci.return_summary_results({'graph':rand_graph, 'val_matrix':np.zeros(rand_graph.shape)})['link_frequency']
    true_linkfreq_boot_mean = true_freq_boot.mean(axis=0)
    true_linkfreq_boot_std = true_freq_boot.std(axis=0)
    del true_freq_boot
     
    results= {  
            #True SCM graph
            "scm_graph":scm_graph,
            # PCMCI+ link frequency and estimated graph, mean and std. of link frequency
            "true_linkfreq": true_linkfreq, "true_graph": true_graph, 
            "true_linkfreq_boot_mean": true_linkfreq_boot_mean,
            "true_linkfreq_boot_std": true_linkfreq_boot_std,
            #Bootstrap-PCMCI+ mean and std. link frequency and estimated graphs
            "boot_linkfreq_mean": boot_linkfreq_mean,"boot_linkfreq_std":boot_linkfreq_std,
            "boot_graphs": boot_graphs
            }

    return results

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
    print("Starting with num_cores = ", num_cores)
    time_start = time.time()
    all_configs = {'results':{}, 
            #True SCM graph
            "scm_graph":{},
            # PCMCI+ link frequency and estimated graph, mean and std. of link frequency
            "true_linkfreq": {}, "true_graph": {}, 
            "true_linkfreq_boot_mean": {},
            "true_linkfreq_boot_std": {},
            #Bootstrap-PCMCI+ mean and std. link frequency and estimated graphs
            "boot_linkfreq_mean": {},"boot_linkfreq_std":{},
            "boot_graphs": {},
            }
    
    job_list = [(i) for i in range(scm_models)]
    num_tasks = scm_models
    num_jobs = max(1,min(num_cores-1, num_tasks))
    def split(a, n):
        k, m = len(a) // n, len(a) % n
        return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

    config_chunks = split(job_list, num_jobs)
    # print config_chunks
    print("num_tasks %s" % num_tasks)
    print("num_jobs %s" % num_jobs)

    ## Send
    for job_id, chunk in enumerate(config_chunks):
        print("submit %d / %d" % (job_id, len(config_chunks)))
        mpi.submit_call("process_chunks", (job_id, chunk), id = job_id)
    ## Retrieve  
    for job_id, chunk in enumerate(config_chunks):
        print("\nreceive %s" % job_id)
        tmp = mpi.get_result(id=job_id)
        for conf_sam in list(tmp.keys()):
            sample = conf_sam
            print(conf_sam)
            all_configs['results'][sample] = tmp[conf_sam]

    print("\nsaving all results...")
    #Gather all results in all_configs dict
    for output_key in all_configs['results'][0].keys():
        all_configs[output_key] = np.zeros((scm_models, ) + all_configs['results'][0][output_key].shape, 
                                            dtype=all_configs['results'][0][output_key].dtype)
        for i in list(all_configs['results'].keys()):
            all_configs[output_key][i] = all_configs['results'][i][output_key]

    del all_configs['results']

    save_path = './' #PATH OF SAVED RESULTS, TO ADJUST
    save_filename = "true_vs_boot_linkfreq_%d-%d-%d-%d-%d-%d-%d-%f.dat" %(scm_models,N,L,T,tau_max,boot_samples,repetions,pc_alpha)

    print("dump ", save_path+save_filename)
    file = open(save_path+save_filename, 'wb')
    pickle.dump(all_configs, file, protocol=-1)        
    file.close()
    time_end = time.time()
    print('Run time in hours ', (time_end - time_start)/3600.)
    return 0
    
mpi.run(verbose=False)


