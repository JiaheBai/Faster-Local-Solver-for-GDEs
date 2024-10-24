import numpy as np
import scipy.sparse as sp
import matplotlib
import warnings
from ppr_solver import *
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="white")
warnings.filterwarnings("ignore")
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"

graph_name = 'wiki-talk'
path = './dataset/'

graph_path = path + graph_name + '/'
adj_matrix = sp.load_npz(graph_path + graph_name + '_csr-mat.npz')
degree = np.array(adj_matrix.sum(1)).flatten()
indices = adj_matrix.indices
indptr = adj_matrix.indptr
all_result = np.load('./results/' + graph_name +'_ppr_exp_ratio_result.npy', allow_pickle=True)
all_result = all_result.reshape(1, -1)[0][0]
sort_all_result = {}
for key in sorted(all_result):
    sort_all_result[key]=all_result[key]
all_result = sort_all_result
indices = list(all_result.keys())
all_result = list(all_result.values())
n = len(all_result)
sor_oper_ratio = np.zeros(n)
sor_algo_time_ratio = np.zeros(n)
appr_sor_algo_time_ratio = np.zeros(n)
appr_sor_oper_ratio = np.zeros(n)
appr_oper_ratio = np.zeros(n)
appr_algo_time_ratio = np.zeros(n)
gd_oper_ratio = np.zeros(n)
gd_algo_time_ratio = np.zeros(n)
local_sor_oper = np.zeros(n)
local_sor_algo_time = np.zeros(n)
global_sor_oper = np.zeros(n)
global_sor_algo_time = np.zeros(n)
local_appr_oper = np.zeros(n)
local_appr_algo_time = np.zeros(n)
global_appr_oper = np.zeros(n)
global_appr_algo_time = np.zeros(n)
local_gd_oper = np.zeros(n)
local_gd_algo_time = np.zeros(n)
global_gd_oper = np.zeros(n)
global_gd_algo_time = np.zeros(n)
for i in range(len(all_result)):
    (local_sor_oper[i], local_sor_algo_time[i], global_sor_oper[i], global_sor_algo_time[i],
         local_appr_oper[i], local_appr_algo_time[i], global_appr_oper[i], global_appr_algo_time[i],
         local_gd_oper[i], local_gd_algo_time[i], global_gd_oper[i], global_gd_algo_time[i]) = all_result[i][1].mean(), all_result[i][2].mean(), all_result[i][3].mean(), all_result[i][4].mean(), all_result[i][5].mean(), all_result[i][6].mean(), all_result[i][7].mean(), all_result[i][8].mean(), all_result[i][9].mean(), all_result[i][10].mean(), all_result[i][11].mean(), all_result[i][12].mean()
for i in range(len(all_result)):
    sor_algo_time_ratio[i] = (all_result[i][4] / all_result[i][2]).mean()
    appr_algo_time_ratio[i] = (all_result[i][8] / all_result[i][6]).mean()
    gd_algo_time_ratio[i] = (all_result[i][12] / all_result[i][10]).mean()
    appr_sor_algo_time_ratio[i] = (all_result[i][6] / all_result[i][2]).mean()
    appr_sor_oper_ratio[i] = (all_result[i][5] / all_result[i][1]).mean()

local_sor_oper_err = np.zeros(n)
local_sor_algo_time_err = np.zeros(n)
global_sor_oper_err = np.zeros(n)
global_sor_algo_time_err = np.zeros(n)
local_appr_oper_err = np.zeros(n)
local_appr_algo_time_err = np.zeros(n)
global_appr_oper_err = np.zeros(n)
global_appr_algo_time_err = np.zeros(n)
local_gd_oper_err = np.zeros(n)
local_gd_algo_time_err = np.zeros(n)
global_gd_oper_err = np.zeros(n)
global_gd_algo_time_err = np.zeros(n)
for i in range(len(all_result)):
    (local_sor_oper_err[i], local_sor_algo_time_err[i], global_sor_oper_err[i], global_sor_algo_time_err[i],
         local_appr_oper_err[i], local_appr_algo_time_err[i], global_appr_oper_err[i], global_appr_algo_time_err[i],
         local_gd_oper_err[i], local_gd_algo_time_err[i], global_gd_oper_err[i], global_gd_algo_time_err[i]) = np.std(all_result[i][1]) / np.sqrt(len(all_result[i][1])), np.std(all_result[i][2]) / np.sqrt(len(all_result[i][2])), np.std(all_result[i][3]) / np.sqrt(len(all_result[i][3])), np.std(all_result[i][4]) / np.sqrt(len(all_result[i][4])), np.std(all_result[i][5]) / np.sqrt(len(all_result[i][5])), np.std(all_result[i][6]) / np.sqrt(len(all_result[i][6])), np.std(all_result[i][7]) / np.sqrt(len(all_result[i][7])), np.std(all_result[i][8]) / np.sqrt(len(all_result[i][8])), np.std(all_result[i][9]) / np.sqrt(len(all_result[i][9])), np.std(all_result[i][10]) / np.sqrt(len(all_result[i][10])), np.std(all_result[i][11]) / np.sqrt(len(all_result[i][11])), np.std(all_result[i][12]) / np.sqrt(len(all_result[i][12]))

gd_algo_time_ratio_err = np.zeros(n)
for i in range(len(all_result)):
    gd_algo_time_ratio_err[i] = np.std(all_result[i][12] / all_result[i][10]) / np.sqrt(len(all_result[i][10]))
fig = plt.figure(figsize=(6.2,5))
ax1 = fig.add_subplot(111)
plt.yscale('log')
plt.xscale('log', base = 2)
ax1.invert_xaxis()
ax2 = ax1.twinx()
plt.yscale('log')
lns1 = ax1.errorbar(indices, local_gd_algo_time, yerr=local_sor_algo_time_err, label='LocalGD', marker='o', linestyle='', markersize=10, elinewidth=2)
lns2 = ax1.errorbar(indices, global_gd_algo_time, yerr=global_sor_algo_time_err, label='GD', marker='v', linestyle='', markersize=10, elinewidth=2)
lns3 = ax2.plot(indices, gd_algo_time_ratio, label='Speedup\nRatio', marker='', linestyle='-', color='darkgreen')
ax2.fill_between(indices, gd_algo_time_ratio - gd_algo_time_ratio_err, gd_algo_time_ratio + gd_algo_time_ratio_err, color='darkgreen',alpha=0.4)
lns4 = ax2.axhline(1, linestyle='--', color = 'limegreen')
lns5 = ax2.axvline(1/(len(indptr) - 1), linestyle='--', color = 'gray')
ytik = [1/(len(indptr) - 1), ]
ytikn = ['$1/n$', ]
for i in range(17, 30, 3):
    ytik.append(np.power(2.0, -i))
    ytikn.append('$2^{-'+str(i)+'}$')
plt.xticks(ytik, ytikn, fontsize=15)
ytik = [1, ]
ytikn = ['$10^0(\\textbf{1})$', ]
for i in range(1, 3):
    ytik.append(np.power(10.0, i))
    ytikn.append('$10^{'+str(i)+'}$')
ax2.set_yticks(ytik, ytikn,fontsize=12)
ax1.set_xlabel('$\epsilon$', fontsize=25)
ax1.set_ylabel('Running Time (s)', fontsize=20)
ax2.set_ylabel('Speedup Ratio', fontsize=20)
lns = [lns1,]+[lns2,]+lns3+[lns4,lns5]
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, fontsize=15, loc=5)
plt.savefig('./figs/'+graph_name+'_ppr_exp_gd_ratio.png', dpi=400, bbox_inches = 'tight')