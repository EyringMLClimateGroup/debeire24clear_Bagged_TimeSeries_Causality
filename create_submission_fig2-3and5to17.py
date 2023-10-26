import os, time, sys
from os import listdir
from os.path import isfile, join
import subprocess
import numpy as np
from random import shuffle
import pickle

import socket

try:
    arg = sys.argv
    submit = int(arg[1])
    print(submit)
    verbosity = 0

except:
    arg = ''
    submit = False
    verbosity = 1
run_locally = False

mypath = './' #PATH OF SAVED RESULTS (to check if results already exist)

num_jobs = 100 #max number of jobs on HPC system
run_time_hrs = 8 #hours of HPC jobs
run_time_min = 0  #min of HPC jobs
num_cpus = 128  #Number of cpu per cores, default 128

samples = 500 #Number of independent models generated

verbosity = 0

anyconfigurations = [] 

fix_metrics = False
overwrite = False


for model in ['random_lineargaussian_highdegree']: # random_linearmixed_highdegree or also random_nonlineargaussian_highdegree
#pc_alpha= [0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01,0.02,0.05,0.1,0.2,0.3, 0.4, 0.5,0.6, 0.8, 0.9, 0.95, 0.98, 0.99, 0.999]

#Indicative values for the parameters T,N and Tau_max for other figures
#highdim = [2,3,5,10,20,30,40]; 
#sample_size=[100,200,500,1000]
#auto=[0., 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 0.95, 0.98, 0.99, 0.999]; 
#tau_max=[5, 10, 15, 20, 25, 30, 35, 40]

  for N in [10]: #[2,3,5,10,20,30,40]
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
      for auto in [0.95]: #[0., 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 0.95, 0.98, 0.99, 0.999]
       for max_true_lag in [5]:
        for contemp_fraction in [0.3]:
          for frac_unobserved in [0.]: #0.3 for LPCMCI experiments
            for T in [200]: #[100,200,500,1000]
              for ci_test in ['par_corr']: 

                method_list = [
                        'standard_pcmci+',
                        'bootstrap_pcmci+',
                        #'pcalg', #for fig 14/15
                        #'bootstrap_pcalg' #for fig 14/15
                        #'lpcmci' #for fig16/17
                        #'bootstrap_lpcmci' #for fig16/17
                        ]
                for method in method_list:
                # below pc_alpha values for Fig3., only [0.001,0.005,0.01,0.02,0.05] is needed for fig 4.
                  for pc_alpha in [0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01,0.02,0.05,0.1,0.2,0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 0.95, 0.98, 0.99, 0.999]:
                   for tau_max in [5]: # [5, 10, 15, 20, 25, 30, 35, 40] for SM experiments
                        if "bootstrap_" not in method:
                            pc_alpha = np.format_float_positional(np.float(pc_alpha),trim='-')
                            para_setup = (model, N, n_links, min_coeff, coeff, auto, contemp_fraction, frac_unobserved, max_true_lag, T, ci_test, method, pc_alpha, tau_max,
                                     0)
                            name = '%s-'*len(para_setup) % para_setup
                            name = name[:-1]
                            anyconfigurations += [name]
                        else:
                            for n_bs in [25,50,100,200]: #number of bootstrap realizations
                                pc_alpha = np.format_float_positional(np.float(pc_alpha),trim='-')
                                para_setup = (model, N, n_links, min_coeff, coeff, auto, contemp_fraction, frac_unobserved, max_true_lag, T, ci_test, method, pc_alpha, tau_max,
                                                        n_bs)
                                name = '%s-'*len(para_setup) % para_setup
                                name = name[:-1]
                                anyconfigurations += [name]


current_results_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]


already_there = []
configurations = []
for conf in anyconfigurations:
    if conf not in configurations:
        conf = conf.replace("'","")

        if (overwrite == False) and (conf + '.dat' in current_results_files):
            already_there.append(conf)
            pass
        else:
            configurations.append(conf)


## CHeck whether configuration already exists!!!
# print configurations
# print already_there
for conf in configurations:
    print(conf)


num_configs = len(configurations)
print("number of todo configs ", num_configs)
print("number of existing configs ", len(already_there))

chunk_length = min(num_jobs, num_configs)
print("num_jobs %s" % num_jobs)
print("chunk_length %s" % chunk_length)
print("cpus %s" % num_cpus)
print("runtime %02.d:%02.d:00" % (run_time_hrs, run_time_min))

print("Shuffle configs to create equal computation time chunks ")
shuffle(configurations)
if num_configs == 0:
    raise ValueError("No configs to do...")

def split(a, n):
    k, m = len(a) // n, len(a) % n
    return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


for config_chunk in split(configurations, chunk_length):

    config_chunk = [con for con in config_chunk if con != None]
    config_string = str(config_chunk)[1:-1].replace(',', '').replace('"', '')

    # 
    job_list = [(conf, i) for i in range(samples) for conf in config_chunk]
    num_tasks = len(config_chunk)*samples
    # print num_tasks
    num_jobs = min(num_cpus-1, num_tasks)

    print(max([len(chunk) for chunk in split(job_list, num_jobs)])) 
    
    use_script = 'compute_fig2-3and5to17.py'

    if submit == False:
        submit_string = ["python", "compute_fig2-3and5to17.py", str(num_cpus), str(samples), str(verbosity)] + config_chunk 

        if run_locally:
            print("Run locally")
            process = subprocess.Popen(submit_string)  #, 
            output = process.communicate()
    if submit:
        submit_string = ['sbatch', '--ntasks', str(num_cpus), '--time', '%02.d:%02.d:00' % (run_time_hrs, run_time_min), 'sbatch_fig2-3and5to17.sh', use_script + " %d %d %d %s" %(num_cpus, samples, verbosity, config_string)]  # +  config_chunk
        print(submit_string[-1])
        process = subprocess.Popen(submit_string)  #, 
        output = process.communicate()
    else:
        print("Not submitted.")

