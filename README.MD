# Bootstrap aggregation and confidence measures to improve time series causal discovery

In this folder, you can find the official implementation of the paper: 
> Debeire, K., Runge, J., Gerhardus, A., Eyring, V. (2023). Bootstrap aggregation and confidence measures to improve time series causal discovery (see [arXiv article](http://arxiv.org/abs/2306.08946))

A method to perform bootstrap aggregation of time series causal graphs and measure a confidence for links of the aggregated graph. Here combined with the PCMCI+ algorithm from the [TIGRAMITE](https://github.com/jakobrunge/tigramite/) package. We provide here the code that can be used to recreate figures 3, 4, and 5 as presented in the paper.

Author: Kevin Debeire, [kevin.debeire@dlr.de](mailto:kevin.debeire@dlr.de)

The current release on Zenodo can be found here: [![DOI](https://zenodo.org/badge/656643051.svg)](https://zenodo.org/badge/latestdoi/656643051)
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

Find below the instructions to produce the **figures 2, 3, 4A, 4B** of the main text and **5 to 17** of the appendix.
**All model and method parameters of the scripts may need to be adjusted to reproduce the figures.**

Please also **adjust the save paths** (by default *./* ) in the scripts.
They specify where the numerical results are saved, and where the figures are saved.

### Generation of the synthetic data:

If you have access to an HPC system with a slurm job scheduler:
- fill in the *sbatch_XXX.sh* scripts. Specify your account, partition, number of cores per CPU, etc...
- **adjust the model parameters and methods** in the *create_submission_figXXX.py* scripts to reproduce a given figure.
- run: 'python create_submission_figXXX.py 1'. This will submit a job for each configuration (for example each alpha_pc of PCMCI+, number of variables, sample size, etc...) in which the Bagged-PCMCI+ and PCMCI+ (or PC/LPCMCI) are evaluated on the generated synthetic data.

Alternatively, it can be run locally, but this is not recommended as the computational cost to run the numerical experiments are high:
- set *run_locally=True* and *submit=False* in *create_submission_figXXX.py*
- **adjust the model parameters and methods** in the *create_submission_figXXX.py* scripts to reproduce a given figure.
- run: 'python create_submission_figXXX.py 1'

**For fig4A**, check the model parameters in compute_fig4A.py and run: 'python compute_fig5A.py'.
**For fig4B**, once all computations are done, run: 'python compute_metrics_fig5B.py'. This will calculate the mean absolute frequencies errors.

### Plotting:
Once the data has been generated, you can plot the figures:
- Use the plotting script matching the figure you want to reproduce.
- You may need to adjust the **values of the model parameters** and **methods** in the plotting scripts.
- For figure 4A, simply run 'python plot_fig4A.py' to reproduce the figure 4A.
- For the other figures, the script expects arguments which describes the type of experiments. For example, run 'python plot_figXXX.py par_corr pc_alpha_highdegree'. The first argument indicates the Conditional Independence test: par_corr (all linear experiments) or gp_dc (nonlinear experiment). The second arguments includes the name of the varying parameters (pc_alpha,sample_size,highdim,tau_max,autocorr) and additional arguments like the degree of cross-links ("highdegree"), the noise distribution ("mixed"), nonlinearity ("nonlinear"), or "boot_rea" for figure 4B. We give a list of the arguments for each figure below.

List of command to run to plot a figure:
- fig2/5 (like fig6/7 and 8/9 and 10/11): 'python plot_fig2and5to17.py par_corr pc_alpha_highdegree' (only the values of the model parameters change)
- fig3a: 'python plot_fig3.py par_corr highdim_highdegree'
- fig3b: 'python plot_fig3.py par_corr sample_size_highdegree'
- fig3c: 'python plot_fig3.py par_corr autocorr_highdegree'
- fig3d: 'python plot_fig3.py par_corr tau_max_highdegree'
- fig4a: 'python plot_fig4A.py'
- fig4b: 'python plot_fig4B.py par_corr boot_rea_highdegree'
- fig12a: 'python plot_fig2and5to17.py par_corr pc_alpha_highdegree_mixed'
- fig12b: 'python plot_fig2and5to17.py par_corr pc_alpha_highdegree_nonlinear'
- fig14/15: 'python plot_fig2and5to17.py par_corr pc_alpha_highdegree' (make sure to change the methods)
- fig16/17: 'python plot_fig2and5to17.py par_corr pc_alpha_highdegree' (make sure to change the methods)

## License
GNU General Public License v3.0

See [license.txt](./license.txt) for full text.

