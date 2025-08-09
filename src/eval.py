from experiment import Query, ExperimentRegistry
from utils import minutes
import ipdb, sys
import yaml
import numpy as np  
import matplotlib.pyplot as plt
import gc
import torch
from sklearn.metrics import roc_auc_score
import seaborn as sns 

exp = ExperimentRegistry(predict_N_times=1)

exp.add_run('stage_0', "/n/data1/hms/dbmi/zaklab/payal/EveryQuery/results/2025-07-22_23-07-53_790919/")
exp.add_run('stage_1', "/n/data1/hms/dbmi/zaklab/payal/EveryQuery/results/2025-07-23_12-01-32_750030/")
exp.add_run('stage_2', "/n/data1/hms/dbmi/zaklab/payal/EveryQuery/results/2025-07-23_17-20-20_211916/")
exp.add_run('stage_3',"/n/data1/hms/dbmi/zaklab/payal/EveryQuery/results/2025-07-22_22-59-42_411916/")

exp.add_run('censor_occurs_separate',"/n/data1/hms/dbmi/zaklab/payal/EveryQuery/results/2025-08-08_23-11-47_886728")

in_queries = exp.get_one_run('stage_0').training_queries

# with open('/home/pac4279/EveryQuery/src/configs/data/codes/hold_out.yaml', 'r') as file:
#     data = yaml.safe_load(file)
#     hold_out_codes = np.random.choice(data, 10, replace=False)\
nans = ['DIAGNOSIS//ICD//9//5941','LAB//51894//UNK','PROCEDURE//ICD//10//0T768DZ','MEDICATION//STOP//Advair Diskus']
out_codes = ['DIAGNOSIS//ICD//10//I82441', 'LAB//224879//UNK', 'DIAGNOSIS//ICD//10//F1020', 'LAB//224434//UNK', 'LAB//224086//UNK', 'HOSPITAL_ADMISSION//AMBULATORY OBSERVATION//PACU', 'MEDICATION//Atropine Sulfate Ophth 1%//Administered', 'MEDICATION//STOP//Nystatin Cream', 'LAB//225764//UNK', 'MEDICATION//STOP//Furosemide','LAB//227073//mEq/L','PROCEDURE//END//225443','DIAGNOSIS//ICD//10//M47896','LAB//227962//UNK','LAB//50905//mg/dL', 'MEDICATION//START//Psyllium', 'DIAGNOSIS//ICD//10//C773', 'LAB//227755//UNK', 'MEDICATION//Piperacillin-Tazobactam//Not Given', 'DRG//HCFA//776//POSTPARTUM & POST ABORTION DIAGNOSES W/O O.R. PROCEDURE', 'LAB//51131//#/uL', 'DIAGNOSIS//ICD//9//7810', 'MEDICATION//START//Thiamine', 'LAB//50957//mmol/L', 'MEDICATION//NIFEdipine (Immediate Release)//Administered',]
out_duration=43200
out_queries = [Query(code=x, duration=out_duration, offset=0, range=None) for x in out_codes]

# for query in out_queries: 
#     print(query)
#     for s in [3,2,1,0]:
#         metric = exp.evaluate(f'stage_{s}', query)['occurs_auc']
#         print(f'\t stage_{s} {metric[0]} ± {metric[1]}')
#     print()

# fig, axes = plt.subplots(2, 2, figsize=(12, 10))
# axes = axes.flatten()
# for s in [0, 1, 2, 3]:
#     stage_training_queries = exp.get_one_run(f'stage_{s}').training_queries
#     for q in stage_training_queries:
#         assert q.code not in out_codes
#         assert q not in out_queries
#     in_aucs = [exp.evaluate(f'stage_{s}', query)['occurs_auc'][0] for query in in_queries]
#     out_aucs = [exp.evaluate(f'stage_{s}', query)['occurs_auc'][0] for query in out_queries]
#     ax = axes[s]
#     ax.hist(in_aucs,  bins=20, alpha=0.7, label='In-distribution')
#     ax.hist(out_aucs, bins=20, alpha=0.5, label='Out-of-distribution')
#     ax.set_title(f"N={len(stage_training_queries)} \n In: {np.mean(in_aucs):.2f}    Out: {np.mean(out_aucs):.2f}")
#     ax.set_xlabel("AUROC")
#     ax.set_ylabel("Count")
#     ax.set_xlim(0,1)
#     ax.set_ylim(0,5)
#     ax.legend(loc='upper left')
# plt.tight_layout()
# plt.savefig('figures/in_out_histogram.png')

predictions = []
run = exp.get_run("/n/data1/hms/dbmi/zaklab/payal/EveryQuery/results/2025-08-08_23-11-47_886728")
for q in list(in_queries): 
    pred = exp.predict(run, q)
    predictions.append((run, q, pred['censor_target'], pred['censor_score'], pred['occurs_target'], pred['occurs_score']))
    del pred
    gc.collect()
    torch.cuda.empty_cache()
fig, axes = plt.subplots(2, 5, figsize=(50, 10))
axes = axes.flatten()
for i in range(len(predictions)): 
    run, q, cen_target, cen_score, occ_target, occ_score = predictions[i]
    occ_score = torch.tensor(occ_score)
    occ_target = torch.tensor(occ_target)
    axes[i].set_title(q.code)
    sns.histplot(occ_score[occ_target==0], alpha=0.5, legend="0", ax=axes[i], stat='density', bins=25)
    sns.histplot(occ_score[occ_target==1], alpha=0.5, legend="1", ax=axes[i], stat='density', bins=10)
plt.legend(); plt.tight_layout(); plt.savefig('figures/occurs_distribution.png')
cen_auc_heatmap = np.zeros((len(predictions),len(predictions)))
occ_auc_heatmap = np.zeros((len(predictions),len(predictions)))
for i, (run_i, q_i, cen_target_i, cen_score_i, occ_target_i, occ_score_i) in enumerate(predictions): 
    for j, (run_j, q_j, cen_target_j, cen_score_j, occ_target_j, occ_score_j) in enumerate(predictions):
        cen_auc_heatmap[i][i] = roc_auc_score(cen_target_i, cen_score_i)
        cen_auc_heatmap[j][j] = roc_auc_score(cen_target_j, cen_score_j)
        cen_auc_heatmap[i][j] = roc_auc_score(cen_target_i, cen_score_j)
        cen_auc_heatmap[j][i] = roc_auc_score(cen_target_j, cen_score_i)
        occ_auc_heatmap[i][i] = roc_auc_score(occ_target_i, occ_score_i)
        occ_auc_heatmap[j][j] = roc_auc_score(occ_target_j, occ_score_j)
        occ_auc_heatmap[i][j] = roc_auc_score(occ_target_i, occ_score_j)
        occ_auc_heatmap[j][i] = roc_auc_score(occ_target_j, occ_score_i)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.heatmap(cen_auc_heatmap*100, ax=axes[0], annot=True, fmt=".4f", annot_kws={"size":8}, cbar=True)
axes[0].set_title("Censor AUC"); axes[0].set_xlabel("Score Query"); axes[0].set_ylabel("Target Query")
sns.heatmap(occ_auc_heatmap*100, ax=axes[1], annot=True, fmt=".0f", annot_kws={"size":8}, cbar=True, vmin=0, vmax=100)
axes[1].set_title("Occurs AUC"); axes[1].set_xlabel("Score Query"); axes[1].set_ylabel("Target Query")
plt.tight_layout(); plt.savefig('figures/permuted_auc.png')
ipdb.set_trace()

exp.plot_auroc_comparison('stage_0','stage_1',in_queries).figure.savefig('figures/in_01.png')
exp.plot_auroc_comparison('stage_0','stage_2',in_queries).figure.savefig('figures/in_02.png')
exp.plot_auroc_comparison('stage_1','stage_2',in_queries).figure.savefig('figures/in_12.png')
exp.plot_auroc_comparison('stage_0','stage_3',in_queries).figure.savefig('figures/in_03.png')
exp.plot_auroc_comparison('stage_1','stage_3',in_queries).figure.savefig('figures/in_13.png')
exp.plot_auroc_comparison('stage_2','stage_3',in_queries).figure.savefig('figures/in_23.png')
exp.plot_auroc_heatmap(['stage_0','stage_1','stage_2', 'stage_3'], in_queries).figure.savefig('figures/in_heatmap.png', dpi=600, bbox_inches="tight")
# exp.plot_auroc_clustermap(['stage_0','stage_1','stage_2', 'stage_3'], eval_queries).figure.savefig('figures/clustermap.png', dpi=600, bbox_inches="tight")

exp.plot_auroc_comparison('stage_0','stage_1',out_queries).figure.savefig('figures/out_01.png')
exp.plot_auroc_comparison('stage_0','stage_2',out_queries).figure.savefig('figures/out_02.png')
exp.plot_auroc_comparison('stage_1','stage_2',out_queries).figure.savefig('figures/out_12.png')
exp.plot_auroc_comparison('stage_0','stage_3',out_queries).figure.savefig('figures/out_03.png')
exp.plot_auroc_comparison('stage_1','stage_3',out_queries).figure.savefig('figures/out_13.png')
exp.plot_auroc_comparison('stage_2','stage_3',out_queries).figure.savefig('figures/out_23.png')
exp.plot_auroc_heatmap(['stage_0','stage_1','stage_2', 'stage_3'], out_queries).figure.savefig('figures/out_heatmap.png', dpi=600, bbox_inches="tight")
