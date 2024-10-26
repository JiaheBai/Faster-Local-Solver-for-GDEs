import numpy as np
import scipy.sparse as sp
import time
import matplotlib
from tqdm import tqdm as tqdm
import warnings
from ppr_solver import *
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="white")
warnings.filterwarnings("ignore")
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"

graph_name = 'ogbn-arxiv'
sor_results = np.load('./results/instantGNN_ogbn-arxiv_sor_result.npz', allow_pickle=True)
fwd_results = np.load('./results/instantGNN_ogbn-arxiv_fwd_result.npz', allow_pickle=True)

sor_results = dict(sor_results)
fwd_results = dict(fwd_results)

sor_pretimes = sor_results['times']
sor_traintimes = sor_results['traintimes']
sor_infertimes = sor_results['infertimes']
sor_train_accs = sor_results['train_accs']
sor_valid_accs = sor_results['valid_accs']
sor_test_accs = sor_results['test_accs']
fwd_pretimes = fwd_results['times']
fwd_traintimes = fwd_results['traintimes']
fwd_infertimes = fwd_results['infertimes']
fwd_train_accs = fwd_results['train_accs']
fwd_valid_accs = fwd_results['valid_accs']
fwd_test_accs = fwd_results['test_accs']
snapshots = len(sor_pretimes)

plt.figure(figsize=(10,5))
bar_width = 0.4
indices = np.arange(snapshots)
plt.bar(indices - bar_width/2, sor_pretimes, bar_width, label='InstantGNN(LocalSOR) Propagation', alpha=0.8)
plt.bar(indices - bar_width/2, sor_traintimes, bar_width, bottom=sor_pretimes, label='InstantGNN(LocalSOR) Training', alpha=0.8)
plt.bar(indices - bar_width/2, sor_infertimes, bar_width, bottom=sor_pretimes+sor_traintimes, label='InstantGNN(LocalSOR) Inference', alpha=0.8)

plt.bar(indices + bar_width/2, fwd_pretimes, bar_width, label='InstantGNN(LocalGS) Propagation', alpha=0.8)
plt.bar(indices + bar_width/2, fwd_traintimes, bar_width, bottom=fwd_pretimes, label='InstantGNN(LocalGS) Training', alpha=0.8)
plt.bar(indices + bar_width/2, fwd_infertimes, bar_width, bottom=fwd_pretimes+fwd_traintimes, label='InstantGNN(LocalGS) Inference', alpha=0.8)

plt.xlabel('Snapshot', fontsize=16)
plt.ylabel('Running Time (s)', fontsize=16)
plt.xticks(indices, indices)
#plt.legend(ncol=2, fontsize=15, bbox_to_anchor=(0.7, 0.99), columnspacing=0.3, loc=4)
plt.legend(ncol=2, fontsize=15, columnspacing=0.3, loc=9)

plt.tight_layout()
plt.savefig('./figs/ogbn-arxiv_instantGNN_runtime.png', dpi=400, bbox_inches = 'tight')
plt.figure()
indices = np.arange(snapshots)
plt.plot(indices, sor_test_accs, marker='o', label='InstantGNN(LocalSOR)', markersize=8)
plt.plot(indices, fwd_test_accs, marker='v', label='InstantGNN(LocalGS)', markersize=8)
plt.xlabel('Snapshot', fontsize=20)
plt.ylabel('Accuracy (\%)', fontsize=22)
plt.legend(fontsize=20, loc=4)
indices = np.array(list(range(5, 16, 5)), dtype=np.int64)
plt.xticks(indices, indices)
indices = np.array(list(range(snapshots)), dtype=np.int64)
plt.xticks(indices, indices, minor=True)

ytik = np.arange(0.67,0.712,0.01)
ytikn = list(range(67, 72))
plt.yticks(ytik, ytikn, fontsize=20)

plt.grid(linestyle='-.', which='major')
plt.tight_layout()
plt.savefig('./figs/ogbn-arxiv_instantGNN_accuracy.png', dpi=400, bbox_inches = 'tight')