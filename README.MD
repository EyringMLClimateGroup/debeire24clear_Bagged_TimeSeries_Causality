# Bootstrap aggregation and confidence measures to improve time series causal discovery

In this folder, you can find the official implementation of the paper: 
> Debeire, K., Runge, J., Gerhardus, A., Eyring, V. (2023). Bootstrap aggregation and confidence measures to improve time series causal discovery (see [arXiv article](http://arxiv.org/abs/2306.08946))

A method to perform bootstrap aggregation of time series causal graphs and measure a confidence for links of the aggregated graph. Here combined with the PCMCI+ algorithm from the [TIGRAMITE](https://github.com/jakobrunge/tigramite/) package. We provide here the code that can be used to recreate figures 3, 4, and 5 as presented in the paper.

Author: Kevin Debeire, [kevin.debeire@dlr.de](mailto:kevin.debeire@dlr.de)

The current release on Zenodo can be found here: [![DOI](url)](url)
## Requirements

First **setup a conda environment** (by default called bagged_pcmci) from the [environment.yml](./environment.yml) file:

```setup
conda env create -f environment.yml
```

**Activate** this environment.

As explained in the paper, our bagging approach and confidence measures is combined with the PCMCI+ causal discovery algorithm. The PCMCI+ method is implemented in the [TIGRAMITE](https://github.com/jakobrunge/tigramite/) package. Follow the instructions below to install TIGRAMITE.

First clone the TIGRAMITE repository:

```
git clone https://github.com/jakobrunge/tigramite.git
```
and point to this specific commit

```
git reset --hard 27ded041e87f8d9a8d5f8714e8db5c1235e8616a
```

Then, our **bagging and confidence measures method** is implemented in the scripts provided in the folder *to_replace_in_tigramite/*.

You will have to replace the TIGRAMITE *pcmci_base.py* and *data_processing.py* files in *tigramite/tigramite* with the ones provided in *to_replace_in_tigramite/*, respectively [*pcmci_base.py*](./to_replace_in_tigramite/pcmci_base.py) and [*data_processing.py*](./to_replace_in_tigramite/data_processing.py).

The modified [*pcmci_base.py*](./to_replace_in_tigramite/pcmci_base.py) and [*data_processing.py*](./to_replace_in_tigramite/data_processing.py) include the bagging and confidence measures introduced in the paper.

Then install TIGRAMITE:

```
python setup.py install
```

 You should now be able to run the numerical experiments and reproduce the figures of the paper.

## Generating numerical experiments data and plotting figures

Find below the instructions to produce the **figure 3, 4, 5A and 5B** of the paper.
**All default model and method parameters of the scripts are already set to reproduce the figures.**

Please **adjust the save paths** (by default *./* ) in the following scripts : [compute_fig4and3.py](./compute_fig4and3.py), [compute_fig5A.py](./compute_fig5A.py), [compute_fig5B.py](./compute_fig5B.py), [compute_metrics_fig5B.py](./compute_metrics_fig5B.py), [plotfig4and3.py](./plot_fig4and3.py), [plot_fig5A.py](./plot_fig5A.py), [plot_fig5B.py](./plot_fig5B.py).
They specify where the numerical results are saved, and where the figures are saved.

For **figure 3 and figure 4**:
- If you have access to an HPC system with slurm job scheduler:
    - fill in [*sbatch_fig4and3.sh*](./sbatch_fig4and3.sh). Specifiy your account, partition, number of cores per CPU, etc...
    - check the model parameters in [*create_submission_fig4and3.py*](./create_submission_fig4and3.py)
    - run: 'python create_submission_fig4and3.py 1'. This will submit a job for each model configurations (here for each alpha_pc of PCMCI+) in which the Bagged-PCMCI+ and PCMCI+ are evaluated on the generated synthetic data 
    - Once all computations are done. Run 'python plot_fig4and3.py par_corr pc_alpha_highdegree'
    - You can modify the values of pc_alpha in 'create_submission_fig4and3.py' and 'plot_fig4and3' to produce figure 4
- Alternatively, it can be run locally, but this is not recommended as the computational cost to run the numerical experiments are high:
    - set *run_locally=True* and *submit=False* in [*create_submission_fig4and3.py*](./create_submission_fig4and3.py)
    - run: 'python create_submission_fig4and3.py 1'
    - Once all computations are done, run 'python plot_fig4and3.py par_corr pc_alpha_highdegree'.


For **figure 5A**:

- If you have access to an HPC system with slurm job scheduler:
    - fill in [*sbatch_fig5A.sh*](./sbatch_fig5A.sh). Specifiy your account, partition, number of cores per CPU, etc...
    - check the model parameters in [*compute_fig5A.py*](./compute_fig5A.py)
    - run: 'sbatch sbatch_fig5A.sh'. This will submit a job for the specified method and model settings. Bagged-PCMCI+ and PCMCI+ are evaluated on generated synthetic data
    - Once all computations are done, run 'python plot_fig5A.py' to reproduce the figure 5A.

- Alternatively, it can be run locally, but this is not recommended as the computational cost to run the numerical experiments are high:
    - check the model parameters in [*compute_fig5A.py*](./compute_fig5A.py)
    - run: 'python compute_fig5A.py' to generate the synthetic data and run Bagged-PCMCI+ and PCMCI+
    - Once all computations are done, run 'python plot_fig5A.py' to reproduce the figure 5A.

For **figure 5B**:

- If you have access to an HPC system with slurm job scheduler:
    - fill in [*sbatch_fig5B.sh*](./sbatch_fig5B.sh). Specifiy your account, partition, number of cores per CPU, etc...
    - check the model parameters in [*create_submission_fig5B.py*](./create_submission_fig5B.py)
    - run: 'python create_submission_fig5B.py 1'. This will submit a job for the specified method and model settings
    - run: 'python compute_metrics_fig5B.py'. This will calculate the mean absolute frequencies errors
    - Run 'python plot_fig5B.py' to reproduce the figure 5A.

- Alternatively, it can be run locally, but this is not recommended as the computational cost to run the numerical experiments are high:
    - set *run_locally=True* and *submit=False* in [*create_submission_fig5B.py*](./create_submission_fig5B.py)
    - run: 'python create_submission_fig5B.py 1'
    - Once all computations are done, run: 'python compute_metrics_fig5B.py'. This will calculate the mean absolute frequencies errors
    - Run 'python plot_fig5B.py par_corr boot_rea_highdegree' to reproduce the figure 5B.

## License
GNU General Public License v3.0

See [license.txt](./license.txt) for full text.

