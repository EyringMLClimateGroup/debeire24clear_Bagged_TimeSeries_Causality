   
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
from matplotlib.ticker import ScalarFormatter, NullFormatter, PercentFormatter
import matplotlib as mpl
params = { 
        'legend.fontsize': 8,
        'legend.handletextpad': .05,
        'lines.color':'black',
        'lines.linewidth':.5,
        'lines.markersize':2,
        'xtick.labelsize':8,
        'xtick.major.pad'  : 1, 
        'xtick.major.size' : 2,
        'ytick.major.pad'  : 1,
        'ytick.major.size' : 2,
        'ytick.labelsize':8,
        'axes.labelsize':8,
        'font.size':10,
        'axes.labelpad':2,
        'axes.spines.right' : False,
        'axes.spines.top' : False,
        }
mpl.rcParams.update(params)

N = 3  #Number of variables
L = 3  #Number of cross links
T = 200 #Time sample size

pc_alpha = 0.01  #alpha_pc of PCMCI+
tau_min = 0
tau_max = 2
boot_samples = 1000  #number of bootstrap realizations B
scm_models = 250   #Number of Structural Causal models
repetions = 100   #Number of indepedendent data samples for each Structural Causal model
auto=0.95 #autocorrelation strength 

#######LOAD RESULTS
save_folder= './' #FOLDER WITH NUMERICAL EXPERIMENTS RESULTS
SAVEFIG_FOLDER = "./" #FOLDER WHERE YOU SAVE THE FIGURES

def cross_line(x,y,xerr,yerr,line_vec=np.array((1,1))):
    #Function to check if point (x,y) with uncertainty (xerr,yerr) crosses 
    #the line starting from the origin with direction defined by line_vec
    vec_xy= np.ma.array((x,y))
    uncertainty_xy= np.ma.array((xerr,yerr),mask = vec_xy.mask)
    ortho_vec = np.array((-line_vec[1], line_vec[0]))/np.linalg.norm(line_vec)
    proj_x = np.dot(vec_xy, ortho_vec) / np.linalg.norm(ortho_vec)
    if abs(proj_x)>np.linalg.norm(uncertainty_xy):
        return 0
    else: return 1

def distance_to_diag(x,y):
        return 100*np.abs(y-x)

cross_line_func = np.vectorize(cross_line, otypes=[int])
dist_func = np.vectorize(distance_to_diag, otypes=[float])


results = pickle.load(open(save_folder+'true_vs_boot_linkfreq_%d-%d-%d-%d-%d-%d-%d-%f.dat' %(scm_models,N,L,T,tau_max,boot_samples,repetions,pc_alpha),"rb"))
print("Loading results from %s" %(save_folder+'true_vs_boot_linkfreq_%d-%d-%d-%d-%d-%d-%d-%f.dat' %(scm_models,N,L,T,tau_max,boot_samples,repetions,pc_alpha)))
true_link_freq = results["true_linkfreq"]
true_links = results["true_graph"]
averaged_boot_linkfreq = results["boot_linkfreq_mean"]
std_boot_linkfreq = results["boot_linkfreq_std"]
boot_links = results["boot_graphs"]
true_freq_boot_mean= results["true_linkfreq_boot_mean"] 
true_freq_boot_std= results["true_linkfreq_boot_std"]
true_link_freq= true_freq_boot_mean
#Define contemp., lagged and any links masks
any_mask = np.zeros((N,N,tau_max+1))
any_mask[:,:,0] = np.eye(N)
any_mask = np.repeat(any_mask.reshape(1, N,N,tau_max + 1), scm_models, axis=0)

contemp_cross_mask_tril = np.ones((N,N,tau_max + 1)).astype('bool')
contemp_cross_mask_tril[:,:,0] = np.eye(N)
contemp_cross_mask_tril = np.repeat(contemp_cross_mask_tril.reshape(1, N,N,tau_max + 1), scm_models, axis=0)

lagged_mask = np.zeros((N,N,tau_max + 1)).astype('bool')
lagged_mask[:,:,0] = 1
lagged_mask = np.repeat(lagged_mask.reshape(1, N,N,tau_max + 1), scm_models, axis=0)

true_links_contempmask = np.ma.array(true_links, mask = contemp_cross_mask_tril)
true_link_freq_contempmask = np.ma.array(true_link_freq, mask = contemp_cross_mask_tril)
averaged_boot_linkfreq_contempmask = np.ma.array(averaged_boot_linkfreq, mask = contemp_cross_mask_tril)

true_links_laggedmask = np.ma.array(true_links, mask = lagged_mask)
true_link_freq_laggedmask = np.ma.array(true_link_freq, mask = lagged_mask)
averaged_boot_linkfreq_laggedmask = np.ma.array(averaged_boot_linkfreq, mask = lagged_mask)

true_links_anymask = np.ma.array(true_links, mask = any_mask)
true_link_freq_anymask = np.ma.array(true_link_freq, mask = any_mask)
averaged_boot_linkfreq_anymask = np.ma.array(averaged_boot_linkfreq, mask = any_mask)
# Plot frequencies across all links of the scm_models different SCMs, each with L links

#########Absent links:
print("Absent links")
print("results over %d true existing links" % true_link_freq_anymask[true_links_anymask!=''].count())
print("results over %d true absent links" % true_link_freq_anymask[true_links_anymask==''].count())
print("Scatter plot should be around the diagonal")
### ALL LINKS
#Frequency of points where the uncertainty bars cross x=y line
n_same_freq = (true_link_freq_anymask[true_links_anymask==''] == averaged_boot_linkfreq_anymask[true_links_anymask=='']).sum()
n_cross_line_absent_links = cross_line_func(x=true_link_freq_anymask[true_links_anymask==''], y=averaged_boot_linkfreq_anymask[true_links_anymask==''], 
                xerr= true_freq_boot_std[true_links_anymask==''], yerr=std_boot_linkfreq[true_links_anymask=='']).sum()
n_cross_line_absent_links_no_xerr = cross_line_func(x=true_link_freq_anymask[true_links_anymask==''], y=averaged_boot_linkfreq_anymask[true_links_anymask==''], 
xerr= 0*true_freq_boot_std[true_links_anymask==''], yerr=std_boot_linkfreq[true_links_anymask=='']).sum()
n_absent_links = true_link_freq_anymask[true_links_anymask==''].count()
mrse = dist_func(true_link_freq_anymask[true_links_anymask==''],averaged_boot_linkfreq_anymask[true_links_anymask=='']).mean()
print("Number of edges with same frequency (1.): %d (%.2f %%)" %(n_same_freq,(100*n_same_freq/n_absent_links)))
print("Frequency of points crossing the x=y line: %.2f %%" %(100*n_cross_line_absent_links/n_absent_links))
print("Frequency of points crossing the x=y line (without xerr): %.2f %%" %(100*n_cross_line_absent_links_no_xerr/n_absent_links))
print("Frequency of points crossing the x=y line (removing same freq. edge): %.2f %%"%(100*(n_cross_line_absent_links-n_same_freq)/(n_absent_links-n_same_freq)))

fig = plt.figure(figsize=(9,5))
ax = fig.add_subplot(231)
ax.errorbar(x=true_link_freq_anymask[true_links_anymask==''], y=averaged_boot_linkfreq_anymask[true_links_anymask==''], xerr= true_freq_boot_std[true_links_anymask==''], yerr=std_boot_linkfreq[true_links_anymask==''],
    color='orange', fmt="o", label='absent links',ecolor='grey',markersize=0.8,elinewidth=0.5)
ax.plot(np.linspace(0.2, 1), np.linspace(0.2, 1), color='black')
ax.legend()
ax.set_title("All absent links")
ax.set_xlabel("Reference link frequency",labelpad=2)
ax.set_ylabel("Bagged-PCMCI+ link frequency",labelpad=2)
ax.set_xticks([0.2*i for i in range(1,6)])
ax.set_yticks([0.2*i for i in range(1,6)])
ax.yaxis.set_major_formatter(PercentFormatter(1))
ax.xaxis.set_major_formatter(PercentFormatter(1))
ax.text(1,0.4,"Num. of links: %d"%n_absent_links,horizontalalignment='right',verticalalignment='top',weight='bold',fontsize=7.5)
ax.text(1,0.25,"Uncertainties crossing x=y %.1f%%"%(100*n_cross_line_absent_links/n_absent_links),
        horizontalalignment='right',verticalalignment='top',weight='bold',fontsize=7.5)
ax.text(1,0.32,"Mean Absolute Error %.3f%%"%mrse,
        horizontalalignment='right',verticalalignment='top',weight='bold',fontsize=7.5)


### CONTEMP LINKS
print("Contemporaneous only")
#Frequency of points where the uncertainty bars cross x=y line
n_same_freq = (true_link_freq_contempmask[true_links_contempmask==''] == averaged_boot_linkfreq_contempmask[true_links_contempmask=='']).sum()
n_cross_line_absent_links = cross_line_func(x=true_link_freq_contempmask[true_links_contempmask==''], y=averaged_boot_linkfreq_contempmask[true_links_contempmask==''], 
                xerr= true_freq_boot_std[true_links_contempmask==''], yerr=std_boot_linkfreq[true_links_contempmask=='']).sum()
n_absent_links = true_link_freq_contempmask[true_links_contempmask==''].count()
mrse = dist_func(true_link_freq_contempmask[true_links_contempmask==''],averaged_boot_linkfreq_contempmask[true_links_contempmask=='']).mean()
print("Number of edges with same frequency (1.): %d (%.2f %%)" %(n_same_freq,100*n_same_freq/n_absent_links))
print("Frequency of points crossing the x=y line: %.2f %%" %(100*n_cross_line_absent_links/n_absent_links))
print("Frequency of points crossing the x=y line (removing same freq. edge): %.2f %%"%(100*(n_cross_line_absent_links-n_same_freq)/(n_absent_links-n_same_freq)))

ax = fig.add_subplot(232)
ax.errorbar(x=true_link_freq_contempmask[true_links_contempmask==''], y=averaged_boot_linkfreq_contempmask[true_links_contempmask==''], 
            xerr= true_freq_boot_std[true_links_contempmask==''], yerr=std_boot_linkfreq[true_links_contempmask==''],
    color='orange', fmt="o", label='absent links',ecolor="grey",markersize=0.8,elinewidth=0.5)
ax.plot(np.linspace(0.2, 1), np.linspace(0.2, 1), color='black')
ax.legend()
ax.set_title("Contemp. absent links")
ax.set_xlabel("Reference link frequency",labelpad=2)
ax.set_ylabel("Bagged-PCMCI+ link frequency",labelpad=2)
ax.set_xticks([0.2*i for i in range(1,6)])
ax.set_yticks([0.2*i for i in range(1,6)])
ax.yaxis.set_major_formatter(PercentFormatter(1))
ax.xaxis.set_major_formatter(PercentFormatter(1))
ax.text(1,0.4,"Num. of links: %d"%n_absent_links,horizontalalignment='right',verticalalignment='top',weight='bold',fontsize=7.5)
ax.text(1,0.25,"Uncertainties crossing x=y %.1f%%"%(100*n_cross_line_absent_links/n_absent_links),
        horizontalalignment='right',verticalalignment='top',weight='bold',fontsize=7.5)
ax.text(1,0.32,"Mean Absolute Error %.3f%%"%mrse,
        horizontalalignment='right',verticalalignment='top',weight='bold',fontsize=7.5)
### LAGGED LINKS
print("Lagged only")
#Frequency of points where the uncertainty bars cross x=y line
n_same_freq = (true_link_freq_laggedmask[true_links_laggedmask==''] == averaged_boot_linkfreq_laggedmask[true_links_laggedmask=='']).sum()
n_cross_line_absent_links = cross_line_func(x=true_link_freq_laggedmask[true_links_laggedmask==''], y=averaged_boot_linkfreq_laggedmask[true_links_laggedmask==''], 
                xerr= true_freq_boot_std[true_links_laggedmask==''], yerr=std_boot_linkfreq[true_links_laggedmask=='']).sum()
n_absent_links = (true_link_freq_laggedmask[true_links_laggedmask=='']).count()
mrse = dist_func(true_link_freq_laggedmask[true_links_laggedmask==''],averaged_boot_linkfreq_laggedmask[true_links_laggedmask=='']).mean()
print("Number of edges with same frequency (1.): %d (%.2f %%)" %(n_same_freq,100*n_same_freq/n_absent_links))
print("Frequency of points crossing the x=y line: %.2f %%" %(100*n_cross_line_absent_links/n_absent_links))
print("Frequency of points crossing the x=y line (removing same freq. edge): %.2f %%"%(100*(n_cross_line_absent_links-n_same_freq)/(n_absent_links-n_same_freq)))

ax = fig.add_subplot(233)
ax.errorbar(x=true_link_freq_laggedmask[true_links_laggedmask==''], y=averaged_boot_linkfreq_laggedmask[true_links_laggedmask==''], 
            xerr= true_freq_boot_std[true_links_laggedmask==''], yerr=std_boot_linkfreq[true_links_laggedmask==''],
    color='orange', fmt="o", label='absent links',ecolor='grey',markersize=0.8,elinewidth=0.5)
ax.plot(np.linspace(0.2, 1), np.linspace(0.2, 1), color='black')
ax.legend()
ax.set_title("Lagged absent links")
ax.set_xlabel("Reference link frequency",labelpad=2)
ax.set_ylabel("Bagged-PCMCI+ link frequency",labelpad=2)
ax.set_xticks([0.2*i for i in range(1,6)])
ax.set_yticks([0.2*i for i in range(1,6)])
ax.yaxis.set_major_formatter(PercentFormatter(1))
ax.xaxis.set_major_formatter(PercentFormatter(1))
ax.text(1,0.4,"Num. of links: %d"%n_absent_links,horizontalalignment='right',verticalalignment='top',weight='bold',fontsize=7.5)
ax.text(1,0.25,"Uncertainties crossing x=y %.1f%%"%(100*n_cross_line_absent_links/n_absent_links),
        horizontalalignment='right',verticalalignment='top',weight='bold',fontsize=7.5)
ax.text(1,0.32,"Mean Absolute Error %.3f%%"%mrse,
        horizontalalignment='right',verticalalignment='top',weight='bold',fontsize=7.5)
######## Existing links plot
print("Existing links")
#Frequency of points where the uncertainty bars cross x=y line
n_same_freq = (true_link_freq_anymask[true_links_anymask!=''] == averaged_boot_linkfreq_anymask[true_links_anymask!='']).sum()
n_cross_line_existing_links = cross_line_func(x=true_link_freq_anymask[true_links_anymask!=''], y=averaged_boot_linkfreq_anymask[true_links_anymask!=''], 
                xerr= true_freq_boot_std[true_links_anymask!=''], yerr=std_boot_linkfreq[true_links_anymask!='']).sum()
n_present_links = true_link_freq_anymask[true_links_anymask!=''].count()
mrse = dist_func(true_link_freq_anymask[true_links_anymask!=''],averaged_boot_linkfreq_anymask[true_links_anymask!='']).mean()
print(n_present_links)
print("Number of edges with same frequency (1.): %d (%.2f %%)" %(n_same_freq,100*n_same_freq/n_present_links))
print("Frequency of points crossing the x=y line: %.2f %%" %(100*n_cross_line_existing_links/n_present_links))
print("Frequency of points crossing the x=y line (removing same freq. edge): %.2f %%"%(100*(n_cross_line_existing_links-n_same_freq)/(n_present_links-n_same_freq)))

ax = fig.add_subplot(234)
ax.errorbar(x=true_link_freq_anymask[true_links_anymask!=''], y=averaged_boot_linkfreq_anymask[true_links_anymask!=''], xerr= true_freq_boot_std[true_links_anymask!=''],yerr=std_boot_linkfreq[true_links_anymask!=''],
    color='orange',  fmt="o", label='existing links',ecolor='grey',markersize=0.8,elinewidth=0.5)
ax.plot(np.linspace(0.2, 1), np.linspace(0.2, 1), color='black')
ax.legend()
ax.set_title("All existing links")
ax.set_xlabel("Reference link frequency",labelpad=2)
ax.set_ylabel("Bagged-PCMCI+ link frequency",labelpad=2)
ax.set_xticks([0.2*i for i in range(1,6)])
ax.set_yticks([0.2*i for i in range(1,6)])
ax.yaxis.set_major_formatter(PercentFormatter(1))
ax.xaxis.set_major_formatter(PercentFormatter(1))
ax.text(1,0.4,"Num. of links: %d"%n_present_links,horizontalalignment='right',verticalalignment='top',weight='bold',fontsize=7.5)
ax.text(1,0.25,"Uncertainties crossing x=y %.1f%%"%(100*n_cross_line_existing_links/n_present_links),
        horizontalalignment='right',verticalalignment='top',weight='bold',fontsize=7.5)
ax.text(1,0.32,"Mean Absolute Error %.3f%%"%mrse,
        horizontalalignment='right',verticalalignment='top',weight='bold',fontsize=7.5)
###Contemp links
print("Contemporaneous only")
#Frequency of points where the uncertainty bars cross x=y line
n_same_freq = (true_link_freq_contempmask[true_links_contempmask!=''] == averaged_boot_linkfreq_contempmask[true_links_contempmask!='']).sum()
n_cross_line_existing_links = cross_line_func(x=true_link_freq_contempmask[true_links_contempmask!=''], y=averaged_boot_linkfreq_contempmask[true_links_contempmask!=''], 
                xerr= true_freq_boot_std[true_links_contempmask!=''], yerr=std_boot_linkfreq[true_links_contempmask!='']).sum()
n_present_links = (true_link_freq_contempmask[true_links_contempmask!='']).count()
mrse = dist_func(true_link_freq_contempmask[true_links_contempmask!=''],averaged_boot_linkfreq_contempmask[true_links_contempmask!='']).mean()
print("Number of edges with same frequency (1.): %d (%.2f %%)" %(n_same_freq,100*n_same_freq/n_present_links))
print("Frequency of points crossing the x=y line: %.2f %%" %(100*n_cross_line_existing_links/n_present_links))
print("Frequency of points crossing the x=y line (removing same freq. edge): %.2f %%"%(100*(n_cross_line_existing_links-n_same_freq)/(n_present_links-n_same_freq)))

ax = fig.add_subplot(235)
ax.errorbar(x=true_link_freq_contempmask[true_links_contempmask!=''], y=averaged_boot_linkfreq_contempmask[true_links_contempmask!=''], 
            xerr= true_freq_boot_std[true_links_contempmask!=''],yerr=std_boot_linkfreq[true_links_contempmask!=''],
    color='orange',  fmt="o", label='existing links',ecolor='grey',markersize=0.8,elinewidth=0.5)
ax.plot(np.linspace(0.2, 1), np.linspace(0.2, 1), color='black')
ax.legend()
ax.set_title("Contemp. existing links")
ax.set_xlabel("Reference link frequency",labelpad=2)
ax.set_ylabel("Bagged-PCMCI+ link frequency",labelpad=2)
ax.set_xticks([0.2*i for i in range(1,6)])
ax.set_yticks([0.2*i for i in range(1,6)])
ax.yaxis.set_major_formatter(PercentFormatter(1))
ax.xaxis.set_major_formatter(PercentFormatter(1))
ax.text(1,0.4,"Num. of links: %d"%n_present_links,horizontalalignment='right',verticalalignment='top',weight='bold',fontsize=7.5)
ax.text(1,0.25,"Uncertainties crossing x=y %.1f%%"%(100*n_cross_line_existing_links/n_present_links),
        horizontalalignment='right',verticalalignment='top',weight='bold',fontsize=7.5)
ax.text(1,0.32,"Mean Absolute Error %.3f%%"%mrse,
        horizontalalignment='right',verticalalignment='top',weight='bold',fontsize=7.5)

### LAGGED LINKS
print("Lagged only")
#Frequency of points where the uncertainty bars cross x=y line
n_same_freq = (true_link_freq_laggedmask[true_links_laggedmask!=''] == averaged_boot_linkfreq_laggedmask[true_links_laggedmask!='']).sum()
n_cross_line_existing_links = cross_line_func(x=true_link_freq_laggedmask[true_links_laggedmask!=''], y=averaged_boot_linkfreq_laggedmask[true_links_laggedmask!=''], 
                xerr= true_freq_boot_std[true_links_laggedmask!=''], yerr=std_boot_linkfreq[true_links_laggedmask!='']).sum()
n_present_links = (true_link_freq_laggedmask[true_links_laggedmask!='']).count()
mrse = dist_func(true_link_freq_laggedmask[true_links_laggedmask!=''],averaged_boot_linkfreq_laggedmask[true_links_laggedmask!='']).mean()
print("Number of edges with same frequency (1.): %d (%.2f %%)" %(n_same_freq,100*n_same_freq/n_present_links))
print("Frequency of points crossing the x=y line: %.2f %%" %(100*n_cross_line_existing_links/n_present_links))
print("Frequency of points crossing the x=y line (removing same freq. edge): %.2f %%"%(100*(n_cross_line_existing_links-n_same_freq)/(n_present_links-n_same_freq)))

ax = fig.add_subplot(236)
ax.errorbar(x=true_link_freq_laggedmask[true_links_laggedmask!=''], y=averaged_boot_linkfreq_laggedmask[true_links_laggedmask!=''], 
            xerr= true_freq_boot_std[true_links_laggedmask!=''],yerr=std_boot_linkfreq[true_links_laggedmask!=''],
    color='orange',  fmt="o", label='existing links',ecolor='grey',markersize=0.8,elinewidth=0.5)
ax.plot(np.linspace(0.2, 1), np.linspace(0.2, 1), color='black')
ax.legend()
ax.set_title("Lagged existing links")
ax.set_xlabel("Reference link frequency",labelpad=2)
ax.set_ylabel("Bagged-PCMCI+ link frequency",labelpad=2)
ax.set_xticks([0.2*i for i in range(1,6)])
ax.set_yticks([0.2*i for i in range(1,6)])
ax.yaxis.set_major_formatter(PercentFormatter(1))
ax.xaxis.set_major_formatter(PercentFormatter(1))
ax.text(1,0.4,"Num. of links: %d"%n_present_links,horizontalalignment='right',verticalalignment='top',weight='bold',fontsize=7.5)
ax.text(1,0.25,"Uncertainties crossing x=y %.1f%%"%(100*n_cross_line_existing_links/n_present_links),
        horizontalalignment='right',verticalalignment='top',weight="bold", fontsize=7.5)
ax.text(1,0.32,"Mean Absolute Error %.3f%%"%mrse,
        horizontalalignment='right',verticalalignment='top',weight='bold',fontsize=7.5)
plt.figtext(0., 1., "A", fontsize=25, fontweight='bold',ha='left', va='top')
fig.subplots_adjust(left=0.08, right=0.98, hspace=.3, bottom=0.10, top=0.90, wspace=.3)

model_name = r"$\mathcal{N}$"
#model_name += r"$_{d{=}1.5}$"
ci_test=r'ParCorr'
plt.figtext(0.98, 0.98, r"%s: $N=%d, T=%d, a=%s$" %(model_name, N, T, auto) 
                        + r", %s, $\alpha=%s, \tau_{\max}=%d$" %(ci_test, pc_alpha, tau_max),
         fontsize=11, ha='right', va='top')

plt.savefig(SAVEFIG_FOLDER+'true_freq_vs_bootstrap_freq_%d_%d_%d_%d_%d_%d_%f.png'
            %(N,T,tau_max,repetions,scm_models,boot_samples,pc_alpha),dpi=600)
print("Saving plot in %s"%(SAVEFIG_FOLDER+'true_freq_vs_bootstrap_freq_%d_%d_%d_%d_%d_%d_%f.png'
            %(N,T,tau_max,repetions,scm_models,boot_samples,pc_alpha)))